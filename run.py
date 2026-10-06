import subprocess
import sys
import time
import urllib.request
from pathlib import Path


PROJECT_ROOT = Path(
    __file__
).resolve().parent


RAW_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
)


MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "best_forecasting_model.joblib"
)


APP_URL = "http://127.0.0.1:8000"


REQUIRED_FILES = [

    "train.csv",

    "test.csv",

    "stores.csv",

    "oil.csv",

    "holidays_events.csv",

    "transactions.csv",

]


def check_dataset():

    missing_files = []

    for file_name in REQUIRED_FILES:

        file_path = (
            RAW_DIR
            / file_name
        )

        if not file_path.exists():

            missing_files.append(
                file_name
            )


    if missing_files:

        print()

        print("=" * 60)

        print(
            "DATASET FILES ARE MISSING"
        )

        print("=" * 60)

        print()

        print(
            "Place the dataset files inside:"
        )

        print(
            RAW_DIR
        )

        print()


        for file_name in missing_files:

            print(
                f"Missing: {file_name}"
            )


        print()

        input(
            "Press Enter to close..."
        )

        return False


    print(
        "All required dataset files found."
    )

    return True


def train_model():

    print()

    print("=" * 60)

    print(
        "STARTING MODEL TRAINING"
    )

    print("=" * 60)

    print()


    pipeline_file = (

        PROJECT_ROOT
        / "src"
        / "pipeline"
        / "train_pipeline.py"

    )


    if not pipeline_file.exists():

        print(
            "Training pipeline not found:"
        )

        print(
            pipeline_file
        )

        input(
            "\nPress Enter to close..."
        )

        return False


    result = subprocess.run(

        [

            sys.executable,

            str(pipeline_file)

        ],

        cwd=PROJECT_ROOT

    )


    if result.returncode != 0:

        print()

        print(
            "Model training failed."
        )

        input(
            "\nPress Enter to close..."
        )

        return False


    if not MODEL_PATH.exists():

        print()

        print(
            "Training completed but model was not created."
        )

        print(
            MODEL_PATH
        )

        input(
            "\nPress Enter to close..."
        )

        return False


    print()

    print(
        "Model training completed successfully."
    )

    print(
        f"Model: {MODEL_PATH}"
    )

    return True


def wait_for_server():

    print()

    print(
        "Waiting for FastAPI server..."
    )


    for _ in range(30):

        try:

            response = (
                urllib.request.urlopen(
                    APP_URL + "/health",
                    timeout=1
                )
            )


            if response.status == 200:

                print(
                    "FastAPI server is ready."
                )

                return True


        except Exception:

            pass


        time.sleep(1)


    print()

    print(
        "FastAPI server did not become ready."
    )

    return False


def open_browser():

    try:

        subprocess.Popen(

            [

                "cmd",

                "/c",

                "start",

                "",

                APP_URL

            ],

            creationflags=(
                subprocess.CREATE_NO_WINDOW
            )

        )


        print(
            f"Opening browser: {APP_URL}"
        )


    except Exception as e:

        print()

        print(
            "Could not open browser automatically."
        )

        print(
            f"Error: {e}"
        )

        print()

        print(
            f"Open manually: {APP_URL}"
        )


def start_api():

    print()

    print("=" * 60)

    print(
        "STARTING REVENUE DEMAND FORECASTING"
    )

    print("=" * 60)

    print()


    api_process = subprocess.Popen(

        [

            sys.executable,

            "-m",

            "uvicorn",

            "app.main:app",

            "--host",

            "127.0.0.1",

            "--port",

            "8000"

        ],

        cwd=PROJECT_ROOT

    )


    server_ready = (
        wait_for_server()
    )


    if not server_ready:

        api_process.terminate()

        input(
            "\nPress Enter to close..."
        )

        return


    open_browser()


    print()

    print("=" * 60)

    print(
        "APPLICATION IS RUNNING"
    )

    print("=" * 60)

    print()

    print(
        f"Dashboard : {APP_URL}"
    )

    print(
        f"Swagger   : {APP_URL}/docs"
    )

    print(
        f"Health    : {APP_URL}/health"
    )

    print()

    print(
        "Press CTRL+C to stop the application."
    )

    print()


    try:

        api_process.wait()


    except KeyboardInterrupt:

        print()

        print(
            "Stopping application..."
        )

        api_process.terminate()


def main():

    print()

    print("=" * 60)

    print(
        "REVENUE DEMAND FORECASTING"
    )

    print("=" * 60)

    print()


    # --------------------------------------------------------
    # 1. Dataset
    # --------------------------------------------------------

    if not check_dataset():

        return


    # --------------------------------------------------------
    # 2. Model
    # --------------------------------------------------------

    if MODEL_PATH.exists():

        print()

        print(
            "Existing trained model found."
        )

        print(
            "Training skipped."
        )


    else:

        print()

        print(
            "Trained model not found."
        )

        print(
            "Starting model training..."
        )


        success = train_model()


        if not success:

            return


    # --------------------------------------------------------
    # 3. Start API
    # --------------------------------------------------------

    start_api()


if __name__ == "__main__":

    main()