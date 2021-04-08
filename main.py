import click
import logging
from face_detector.face_detector import raw_images_processing
from face_detector.face_identifier import predict_input_identity


LOGGER = logging.getLogger(__name__)


@click.group()
def cli():
    pass

@cli.command()
def apply_face_detection_on_benchmark_people():
    LOGGER.warning("Running Face detection on benchmark data and store results in the filesystem...")
    raw_images_processing()
    LOGGER.warning("Extraction done")


@cli.command()
def make_face_identification():
    """
    Command to run a face recognition without using the web server.
    :return:
    """
    predict_input_identity()


if __name__ == "__main__":
    cli()
