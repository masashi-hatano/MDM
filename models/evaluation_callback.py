import os
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

# Import your evaluation functions
# from eval_humanml import evaluate_matching_score, evaluate_fid, evaluate_diversity


class HumanMLEvaluationCallback(pl.Callback):
    """
    Custom callback for comprehensive HumanML3D evaluation during training
    """

    def __init__(
        self,
        eval_dataloader=None,
        eval_every_n_epochs=10,
        num_samples=1024,
        save_eval_results=True,
        eval_output_dir="eval_results",
    ):
        super().__init__()
        self.eval_dataloader = eval_dataloader
        self.eval_every_n_epochs = eval_every_n_epochs
        self.num_samples = num_samples
        self.save_eval_results = save_eval_results
        self.eval_output_dir = Path(eval_output_dir)
        self.eval_output_dir.mkdir(exist_ok=True)

    def on_validation_epoch_end(self, trainer, pl_module):
        """Run comprehensive evaluation at the end of validation epoch"""
        if trainer.current_epoch % self.eval_every_n_epochs != 0:
            return

        print(
            f"\n========== Running Comprehensive Evaluation at Epoch {trainer.current_epoch} =========="
        )

        # Set model to eval mode
        pl_module.eval()

        # Run evaluation
        results = self.run_evaluation(trainer, pl_module)

        # Log results
        self.log_results(trainer, pl_module, results)

        # Save results if requested
        if self.save_eval_results:
            self.save_results(trainer, results)

    def run_evaluation(self, trainer, pl_module):
        """Run the actual evaluation"""
        results = OrderedDict()

        # Generate samples for evaluation
        print("Generating samples...")
        generated_samples, text_prompts = self.generate_evaluation_samples(pl_module)

        # Here you would integrate your evaluation functions from eval_humanml.py
        # For now, we'll use placeholder values

        # Placeholder evaluation - replace with actual eval_humanml.py functions
        results["fid"] = np.random.uniform(10, 100)  # Placeholder
        results["r_precision_top1"] = np.random.uniform(0.3, 0.7)  # Placeholder
        results["r_precision_top2"] = np.random.uniform(0.4, 0.8)  # Placeholder
        results["r_precision_top3"] = np.random.uniform(0.5, 0.9)  # Placeholder
        results["diversity"] = np.random.uniform(5, 15)  # Placeholder
        results["multimodality"] = np.random.uniform(1, 5)  # Placeholder

        # TODO: Replace placeholders with actual evaluation code:
        # results = self.run_actual_evaluation(generated_samples, text_prompts)

        return results

    def generate_evaluation_samples(self, pl_module):
        """Generate samples for evaluation"""
        # Sample text prompts from your dataset
        text_prompts = [
            "a person is walking forward",
            "a person is jumping",
            "a person is dancing",
            "a person is running",
            "a person is sitting down",
        ] * (
            self.num_samples // 5
        )  # Repeat to get desired number of samples

        # Generate samples
        with torch.no_grad():
            samples = pl_module.generate_samples(text_prompts[: self.num_samples])

        return samples, text_prompts[: self.num_samples]

    def run_actual_evaluation(self, generated_samples, text_prompts):
        """
        Run actual evaluation using your eval_humanml.py functions
        TODO: Integrate your evaluation code here
        """
        # This is where you'd call your evaluation functions:
        #
        # # Prepare data loaders
        # motion_loaders = self.prepare_motion_loaders(generated_samples, text_prompts)
        #
        # # Run evaluations
        # match_scores, r_precision, activations = evaluate_matching_score(eval_wrapper, motion_loaders, file)
        # fid_scores = evaluate_fid(eval_wrapper, gt_loader, activations, file)
        # diversity_scores = evaluate_diversity(activations, file, diversity_times)
        #
        # return {
        #     'fid': fid_scores['generated'],
        #     'r_precision_top1': r_precision['generated'][0],
        #     'r_precision_top2': r_precision['generated'][1],
        #     'r_precision_top3': r_precision['generated'][2],
        #     'diversity': diversity_scores['generated'],
        # }

        pass  # Placeholder

    def log_results(self, trainer, pl_module, results):
        """Log evaluation results to tensorboard/wandb"""
        for metric_name, value in results.items():
            pl_module.log(f"eval/{metric_name}", value, on_epoch=True)
            print(f"eval/{metric_name}: {value:.4f}")

    def save_results(self, trainer, results):
        """Save evaluation results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = (
            self.eval_output_dir
            / f"eval_results_epoch_{trainer.current_epoch}_{timestamp}.txt"
        )

        with open(filename, "w") as f:
            f.write(f"Evaluation Results - Epoch {trainer.current_epoch}\n")
            f.write(f"Timestamp: {datetime.now()}\n")
            f.write("=" * 50 + "\n")

            for metric_name, value in results.items():
                f.write(f"{metric_name}: {value:.4f}\n")

        print(f"Evaluation results saved to: {filename}")
