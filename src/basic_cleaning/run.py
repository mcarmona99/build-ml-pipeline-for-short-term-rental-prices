#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
import pandas as pd
import wandb

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    logger.info("Downloading input artifact")
    artifact_local_path = run.use_artifact(args.input_artifact).file()

    logger.info("Reading downloaded artifact with pandas")
    df = pd.read_csv(artifact_local_path)

    logger.info("Dropping outliers")
    idx = df['price'].between(args.min_price, args.max_price)
    df = df[idx].copy()

    logger.info("Converting last_review to datetime")
    df['last_review'] = pd.to_datetime(df['last_review'])

    logger.info("Saving cleaned df and uploading to wandb")
    df.to_csv("clean_sample.csv", index=False)

    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file("clean_sample.csv")

    logger.info("Logging artifacts and finishing wandb run")
    run.log_artifact(artifact)
    run.finish()

    logger.info("Step finished")


if __name__ == "__main__":
    logger.info("Setting parser with the command line arguments")

    parser = argparse.ArgumentParser(description="A very basic data cleaning")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Input artifact name",
        required=True
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Output artifact",
        required=True
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help="Output type",
        required=True
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="Output description",
        required=True
    )

    parser.add_argument(
        "--min_price",
        type=float,
        help="Float indicating the minimum price",
        required=True
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help="Float indicating the maximum price",
        required=True
    )

    args = parser.parse_args()

    logger.info("Starting wandb instance")

    go(args)
