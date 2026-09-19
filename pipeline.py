from pathlib import Path
from data_ingestion import DataIngestion
from train import CreditScoreTrainer
from evaluation import CreditScoreEvaluator

class CreditScorePipeline:
    def __init__(self, input_path: str, output_dir: str = "artifacts"):
        self.input_path = input_path
        self.output_dir = output_dir

        self.ingestor = DataIngestion(input_path=self.input_path, output_dir=self.output_dir)
        self.trainer = CreditScoreTrainer()
        self.evaluator = CreditScoreEvaluator()

    def execute(self):
        print("Starting Credit Score Pipeline...")

        # 1. Data Ingestion
        data_path = self.ingestor.run()

        # 2. Training
        models, run_ids, x_test, y_test = self.trainer.run(data_path)

        # 3. Evaluation
        results = self.evaluator.evaluate(models, run_ids, x_test, y_test)
        print("\n=== Evaluation Results ===")
        print(results)

        return results

if __name__ == "__main__":
    DATA_INPUT = Path(__file__).parent / "data_B.csv"

    pipeline = CreditScorePipeline(input_path=DATA_INPUT, output_dir="artifacts")
    pipeline.execute()