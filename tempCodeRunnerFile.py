from src.utils import create_directories, setup_logger


def main():

    create_directories()

    logger = setup_logger()

    logger.info("Starting Forecasting Project")

    logger.info("Directories created successfully")

    print("Project setup completed successfully")


if __name__ == "__main__":
    main()