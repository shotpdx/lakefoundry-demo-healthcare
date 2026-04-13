#!/usr/bin/env python3
"""
Deploy the diabetic outcomes DLT pipeline to Databricks.

This script uploads the transformation files and creates/updates the pipeline.
Run this from the project root directory.

Usage:
    python scripts/deploy_pipeline.py
"""

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.pipelines import PipelineLibrary, NotebookLibrary
import os

# Configuration
PIPELINE_NAME = "[dev] diabetic_outcomes_pipeline"
CATALOG = "lakefoundry_dev"
SCHEMA = "hls_demo_omop_analytics"
WORKSPACE_PATH = "/Workspace/Shared/lakefoundry/diabetic_outcomes"

# Transformation files
TRANSFORMATION_FILES = [
    "src/diabetic_outcomes/transformations/diabetic_cohort_summary.py",
    "src/diabetic_outcomes/transformations/survival_statistics.py",
]


def main():
    w = WorkspaceClient()
    
    # Upload transformation files
    print(f"Uploading transformation files to {WORKSPACE_PATH}...")
    for local_path in TRANSFORMATION_FILES:
        filename = os.path.basename(local_path)
        workspace_file_path = f"{WORKSPACE_PATH}/transformations/{filename}"
        
        with open(local_path, "rb") as f:
            content = f.read()
        
        # Create directory if needed
        try:
            w.workspace.mkdirs(f"{WORKSPACE_PATH}/transformations")
        except Exception:
            pass  # Directory may already exist
        
        # Upload file
        w.workspace.upload(
            workspace_file_path,
            content,
            overwrite=True,
            format="SOURCE"
        )
        print(f"  Uploaded: {workspace_file_path}")
    
    # Create or update pipeline
    print(f"\nCreating/updating pipeline: {PIPELINE_NAME}...")
    
    libraries = [
        PipelineLibrary(
            notebook=NotebookLibrary(
                path=f"{WORKSPACE_PATH}/transformations/{os.path.basename(f)}"
            )
        )
        for f in TRANSFORMATION_FILES
    ]
    
    # Check if pipeline exists
    existing = None
    for p in w.pipelines.list_pipelines():
        if p.name == PIPELINE_NAME:
            existing = p
            break
    
    if existing:
        print(f"  Updating existing pipeline: {existing.pipeline_id}")
        w.pipelines.update(
            pipeline_id=existing.pipeline_id,
            name=PIPELINE_NAME,
            catalog=CATALOG,
            target=SCHEMA,
            libraries=libraries,
            serverless=True,
            development=True,
            continuous=False,
            channel="CURRENT",
        )
        pipeline_id = existing.pipeline_id
    else:
        print("  Creating new pipeline...")
        result = w.pipelines.create(
            name=PIPELINE_NAME,
            catalog=CATALOG,
            target=SCHEMA,
            libraries=libraries,
            serverless=True,
            development=True,
            continuous=False,
            channel="CURRENT",
        )
        pipeline_id = result.pipeline_id
        print(f"  Created pipeline: {pipeline_id}")
    
    # Start pipeline update
    print("\nStarting pipeline update...")
    update = w.pipelines.start_update(pipeline_id=pipeline_id)
    print(f"  Update started: {update.update_id}")
    
    # Wait for completion
    print("  Waiting for completion...")
    import time
    while True:
        status = w.pipelines.get_update(pipeline_id=pipeline_id, update_id=update.update_id)
        state = status.update.state.value if status.update.state else "UNKNOWN"
        print(f"    State: {state}")
        
        if state in ("COMPLETED", "FAILED", "CANCELED"):
            break
        
        time.sleep(10)
    
    if state == "COMPLETED":
        print("\nPipeline update completed successfully!")
        print(f"\nOutput tables:")
        print(f"  - {CATALOG}.{SCHEMA}.diabetic_cohort_summary")
        print(f"  - {CATALOG}.{SCHEMA}.survival_statistics")
    else:
        print(f"\nPipeline update {state}. Check the Databricks UI for details.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
