import os
import time
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tensorflow as tf
from datetime import datetime
import logging
import pandas as pd
from pathlib import Path


class AeroRLLogger:
    def __init__(self, config_path="config/config.yaml", use_tensorboard=True, use_matplotlib=True):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Set up log directories
        self.base_log_dir = self.config['paths']['logs_dir']
        self.run_id = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.log_dir = os.path.join(self.base_log_dir, self.run_id)

        # Create necessary directories
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(os.path.join(self.log_dir, 'tensorboard'), exist_ok=True)
        os.makedirs(os.path.join(self.log_dir, 'plots'), exist_ok=True)
        os.makedirs(os.path.join(self.log_dir, 'cfd_images'), exist_ok=True)
        os.makedirs(os.path.join(self.log_dir, 'models'), exist_ok=True)

        # Set up Python logging
        self._setup_file_logging()

        # Initialize visualization tools
        self.use_tensorboard = use_tensorboard
        self.use_matplotlib = use_matplotlib

        if self.use_tensorboard:
            self.tb_writer = tf.summary.create_file_writer(
                os.path.join(self.log_dir, 'tensorboard')
            )

        if self.use_matplotlib:
            self.metrics_history = {
                'iteration': [],
                'timestep': [],
                'episode': [],
                'total_reward': [],
                'episode_length': [],
                'drag_coefficient': [],
                'downforce_coefficient': [],
                'efficiency': [],
                'loss': [],
                'value_loss': [],
                'policy_loss': [],
                'entropy': [],
                'learning_rate': [],
            }

        # Save the configuration file for this run
        self._save_config()

        # Log the initialization
        logging.info(f"Logger initialized with run ID: {self.run_id}")
        logging.info(f"Log directory: {self.log_dir}")

        # Print initialization message
        print(f"AeroRL Logger initialized - Run ID: {self.run_id}")
        print(f"Log directory: {self.log_dir}")

    def _setup_file_logging(self):
        """Set up file-based logging."""
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        log_file = os.path.join(self.log_dir, 'aerorl.log')

        # Configure root logger
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

    def _save_config(self):
        """Save the configuration used for this run."""
        config_path = os.path.join(self.log_dir, 'config_used.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        logging.info(f"Configuration saved to {config_path}")

    def log_scalar(self, tag, value, step):
        if self.use_tensorboard:
            with self.tb_writer.as_default():
                tf.summary.scalar(tag, value, step=step)

        if self.use_matplotlib:
            # Only add to metrics_history if the tag exists
            if tag in self.metrics_history:
                self.metrics_history[tag].append(value)

                # Ensure the iteration is also recorded
                if tag != 'iteration' and 'iteration' in self.metrics_history:
                    if len(self.metrics_history['iteration']) < len(self.metrics_history[tag]):
                        self.metrics_history['iteration'].append(step)

        # Also log to standard logger at appropriate intervals
        if step % self.config['visualization']['plot_frequency'] == 0:
            logging.info(f"Step {step}: {tag} = {value}")

    def log_histogram(self, tag, values, step):
        if self.use_tensorboard:
            with self.tb_writer.as_default():
                tf.summary.histogram(tag, values, step=step)

    def log_cfd_image(self, tag, image_path, step):
        # Save a copy of the image in the log directory
        if os.path.exists(image_path):
            dest_path = os.path.join(
                self.log_dir, 'cfd_images', f"{tag}_{step}.png")
            tf.io.gfile.copy(image_path, dest_path, overwrite=True)

            # Log to TensorBoard if enabled
            if self.use_tensorboard:
                image = tf.io.read_file(image_path)
                image = tf.image.decode_png(image, channels=3)
                image = tf.expand_dims(image, 0)  # Add batch dimension
                with self.tb_writer.as_default():
                    tf.summary.image(tag, image, step=step)

            logging.info(f"Saved CFD image for step {step}: {dest_path}")
        else:
            logging.warning(f"Image not found at {image_path}")

    def log_episode(self, episode_num, timestep, total_reward, episode_length,
                    drag_coef, downforce_coef, additional_metrics=None):
        # Calculate efficiency (downforce-to-drag ratio)
        efficiency = abs(downforce_coef / drag_coef) if drag_coef != 0 else 0

        # Log to TensorBoard
        if self.use_tensorboard:
            with self.tb_writer.as_default():
                tf.summary.scalar('episode/total_reward',
                                  total_reward, step=episode_num)
                tf.summary.scalar('episode/length',
                                  episode_length, step=episode_num)
                tf.summary.scalar('aerodynamics/drag_coefficient',
                                  drag_coef, step=episode_num)
                tf.summary.scalar(
                    'aerodynamics/downforce_coefficient', downforce_coef, step=episode_num)
                tf.summary.scalar('aerodynamics/efficiency',
                                  efficiency, step=episode_num)

                # Log any additional metrics
                if additional_metrics:
                    for key, value in additional_metrics.items():
                        tf.summary.scalar(
                            f'episode/{key}', value, step=episode_num)

        # Log to matplotlib history if enabled
        if self.use_matplotlib:
            self.metrics_history['episode'].append(episode_num)
            self.metrics_history['timestep'].append(timestep)
            self.metrics_history['total_reward'].append(total_reward)
            self.metrics_history['episode_length'].append(episode_length)
            self.metrics_history['drag_coefficient'].append(drag_coef)
            self.metrics_history['downforce_coefficient'].append(
                downforce_coef)
            self.metrics_history['efficiency'].append(efficiency)

        # Log to standard logger
        logging.info(f"Episode {episode_num} completed:")
        logging.info(f"  Total reward: {total_reward:.4f}")
        logging.info(f"  Episode length: {episode_length}")
        logging.info(f"  Drag coefficient: {drag_coef:.4f}")
        logging.info(f"  Downforce coefficient: {downforce_coef:.4f}")
        logging.info(f"  Efficiency (downforce/drag): {efficiency:.4f}")

        # Save episode data to CSV
        self._save_episode_data_to_csv(episode_num, timestep, total_reward,
                                       episode_length, drag_coef, downforce_coef,
                                       efficiency, additional_metrics)

    def _save_episode_data_to_csv(self, episode_num, timestep, total_reward,
                                  episode_length, drag_coef, downforce_coef,
                                  efficiency, additional_metrics=None):
        episode_data = {
            'episode': episode_num,
            'timestep': timestep,
            'total_reward': total_reward,
            'episode_length': episode_length,
            'drag_coefficient': drag_coef,
            'downforce_coefficient': downforce_coef,
            'efficiency': efficiency,
        }

        # Add any additional metrics
        if additional_metrics:
            episode_data.update(additional_metrics)

        # Convert to DataFrame and save
        df = pd.DataFrame([episode_data])
        csv_path = os.path.join(self.log_dir, 'episode_data.csv')

        # Append to existing file or create new one
        if os.path.exists(csv_path):
            df.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            df.to_csv(csv_path, index=False)

    def log_training_metrics(self, step, loss, value_loss=None, policy_loss=None,
                             entropy=None, learning_rate=None):
        # Log to TensorBoard
        if self.use_tensorboard:
            with self.tb_writer.as_default():
                tf.summary.scalar('training/loss', loss, step=step)

                if value_loss is not None:
                    tf.summary.scalar('training/value_loss',
                                      value_loss, step=step)

                if policy_loss is not None:
                    tf.summary.scalar('training/policy_loss',
                                      policy_loss, step=step)

                if entropy is not None:
                    tf.summary.scalar('training/entropy', entropy, step=step)

                if learning_rate is not None:
                    tf.summary.scalar('training/learning_rate',
                                      learning_rate, step=step)

        # Log to matplotlib history if enabled
        if self.use_matplotlib:
            self.metrics_history['iteration'].append(step)
            self.metrics_history['loss'].append(loss)

            if value_loss is not None:
                self.metrics_history['value_loss'].append(value_loss)

            if policy_loss is not None:
                self.metrics_history['policy_loss'].append(policy_loss)

            if entropy is not None:
                self.metrics_history['entropy'].append(entropy)

            if learning_rate is not None:
                self.metrics_history['learning_rate'].append(learning_rate)

        # Log periodic updates
        if step % self.config['visualization']['plot_frequency'] == 0:
            logging.info(f"Training step {step}:")
            logging.info(f"  Loss: {loss:.6f}")

            if value_loss is not None:
                logging.info(f"  Value loss: {value_loss:.6f}")

            if policy_loss is not None:
                logging.info(f"  Policy loss: {policy_loss:.6f}")

            if entropy is not None:
                logging.info(f"  Entropy: {entropy:.6f}")

            if learning_rate is not None:
                logging.info(f"  Learning rate: {learning_rate:.6f}")

    def save_model_checkpoint(self, model, step, is_best=False):
        checkpoint_dir = os.path.join(self.log_dir, 'models')
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save the model
        if hasattr(model, 'save_weights'):
            # TensorFlow-style model
            checkpoint_path = os.path.join(
                checkpoint_dir, f"model_step_{step}")
            model.save_weights(checkpoint_path)
            logging.info(
                f"Model checkpoint saved at step {step}: {checkpoint_path}")

            # Save best model separately
            if is_best:
                best_path = os.path.join(checkpoint_dir, "best_model")
                model.save_weights(best_path)
                logging.info(f"Best model saved at step {step}: {best_path}")
        elif hasattr(model, 'state_dict'):
            # PyTorch-style model
            checkpoint_path = os.path.join(
                checkpoint_dir, f"model_step_{step}.pt")
            torch.save(model.state_dict(), checkpoint_path)
            logging.info(
                f"Model checkpoint saved at step {step}: {checkpoint_path}")

            # Save best model separately
            if is_best:
                best_path = os.path.join(checkpoint_dir, "best_model.pt")
                torch.save(model.state_dict(), best_path)
                logging.info(f"Best model saved at step {step}: {best_path}")
        else:
            logging.warning(
                "Model format not recognized, checkpoint not saved.")

    def plot_training_curves(self, save=True, show=False):
        if not self.use_matplotlib or not self.metrics_history['iteration']:
            logging.warning("No matplotlib data available for plotting.")
            return

        # Create figure directory if it doesn't exist
        plot_dir = os.path.join(self.log_dir, 'plots')
        os.makedirs(plot_dir, exist_ok=True)

        # Plot 1: Training losses
        if self.metrics_history.get('loss'):
            plt.figure(figsize=(12, 8))
            plt.plot(self.metrics_history['iteration'],
                     self.metrics_history['loss'], label='Total Loss')

            if self.metrics_history.get('value_loss'):
                plt.plot(
                    self.metrics_history['iteration'], self.metrics_history['value_loss'], label='Value Loss')

            if self.metrics_history.get('policy_loss'):
                plt.plot(
                    self.metrics_history['iteration'], self.metrics_history['policy_loss'], label='Policy Loss')

            plt.xlabel('Training Step')
            plt.ylabel('Loss')
            plt.title('Training Losses')
            plt.legend()
            plt.grid(True)

            if save:
                plt.savefig(os.path.join(
                    plot_dir, 'training_losses.png'), dpi=300)

            if show:
                plt.show()
            else:
                plt.close()

        # Plot 2: Rewards per episode
        if self.metrics_history.get('episode') and self.metrics_history.get('total_reward'):
            plt.figure(figsize=(12, 8))
            plt.plot(self.metrics_history['episode'],
                     self.metrics_history['total_reward'])
            plt.xlabel('Episode')
            plt.ylabel('Total Reward')
            plt.title('Reward per Episode')
            plt.grid(True)

            if save:
                plt.savefig(os.path.join(
                    plot_dir, 'episode_rewards.png'), dpi=300)

            if show:
                plt.show()
            else:
                plt.close()

        # Plot 3: Aerodynamic coefficients
        if (self.metrics_history.get('episode') and
            self.metrics_history.get('drag_coefficient') and
                self.metrics_history.get('downforce_coefficient')):

            fig, ax1 = plt.subplots(figsize=(12, 8))

            # Drag coefficient (positive values)
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Drag Coefficient', color='tab:red')
            ax1.plot(self.metrics_history['episode'], self.metrics_history['drag_coefficient'],
                     color='tab:red', label='Drag')
            ax1.tick_params(axis='y', labelcolor='tab:red')

            # Downforce coefficient (negative values typically)
            ax2 = ax1.twinx()
            ax2.set_ylabel('Downforce Coefficient', color='tab:blue')
            ax2.plot(self.metrics_history['episode'], self.metrics_history['downforce_coefficient'],
                     color='tab:blue', label='Downforce')
            ax2.tick_params(axis='y', labelcolor='tab:blue')

            # Add legend
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

            plt.title('Aerodynamic Coefficients over Episodes')
            plt.grid(True)

            if save:
                plt.savefig(os.path.join(
                    plot_dir, 'aero_coefficients.png'), dpi=300)

            if show:
                plt.show()
            else:
                plt.close()

        # Plot 4: Efficiency (downforce/drag ratio)
        if self.metrics_history.get('episode') and self.metrics_history.get('efficiency'):
            plt.figure(figsize=(12, 8))
            plt.plot(self.metrics_history['episode'],
                     self.metrics_history['efficiency'])
            plt.xlabel('Episode')
            plt.ylabel('Efficiency (Downforce/Drag)')
            plt.title('Aerodynamic Efficiency over Episodes')
            plt.grid(True)

            if save:
                plt.savefig(os.path.join(plot_dir, 'efficiency.png'), dpi=300)

            if show:
                plt.show()
            else:
                plt.close()

        logging.info(f"Training curves plotted and saved to {plot_dir}")

    def close(self):
        if self.use_matplotlib:
            self.plot_training_curves(save=True, show=False)

        # Export metrics to CSV for further analysis
        if self.metrics_history and self.metrics_history.get('iteration'):
            try:
                # Convert metrics to DataFrame
                metrics_df = pd.DataFrame(self.metrics_history)
                metrics_df.to_csv(os.path.join(
                    self.log_dir, 'training_metrics.csv'), index=False)
                logging.info(f"Training metrics exported to CSV")
            except Exception as e:
                logging.error(f"Error exporting metrics to CSV: {e}")

        if self.use_tensorboard:
            self.tb_writer.close()

        logging.info(f"Logger closed - Run ID: {self.run_id}")
        print(f"AeroRL Logger closed - Run ID: {self.run_id}")
        print(f"Results saved to: {self.log_dir}")


# Utility functions for the logger
def create_progress_bar(iterable, total=None, desc="Progress", ncols=100):
    try:
        from tqdm import tqdm
        return tqdm(iterable, total=total, desc=desc, ncols=ncols)
    except ImportError:
        logging.warning("tqdm not installed. Progress bar not available.")
        return iterable


def visualize_cfd_results(pressure_data, velocity_data, output_path, title="CFD Results"):
    try:
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Plot pressure field
        c1 = ax1.imshow(pressure_data, cmap='jet')
        ax1.set_title('Pressure Field')
        plt.colorbar(c1, ax=ax1, label='Pressure (Pa)')

        # Plot velocity magnitude
        velocity_magnitude = np.sqrt(np.sum(velocity_data**2, axis=-1))
        c2 = ax2.imshow(velocity_magnitude, cmap='viridis')
        ax2.set_title('Velocity Field')
        plt.colorbar(c2, ax=ax2, label='Velocity (m/s)')

        # Set main title
        plt.suptitle(title)

        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save figure
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        return output_path
    except Exception as e:
        logging.error(f"Error creating CFD visualization: {e}")
        return None


# Create a singleton logger instance
_logger_instance = None


def get_logger(config_path="config/config.yaml", use_tensorboard=True, use_matplotlib=True):
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = AeroRLLogger(
            config_path=config_path,
            use_tensorboard=use_tensorboard,
            use_matplotlib=use_matplotlib
        )
    return _logger_instance
