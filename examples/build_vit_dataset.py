import argparse
import io
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import PIL
import requests
from bigxml import BigXmlError, Parser, xml_handle_element
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_product_id(url: str) -> str | None:
    parsed_url = urlparse(url)
    id_part = parsed_url.path.split("_id")[-1][:-1]
    if id_part:
        return id_part
    return None


def extract_good_cod(filename: str) -> str | None:
    pattern = r"\b([89]\d{7,})(?=\D|$)"  # the first digit of a good_cod 8 or higher
    match = re.search(pattern, filename)

    if match:
        good_cod = match.group(1)
        return good_cod
    return None


def fix_amp(iterable):
    iterator = iter(iterable)
    for data in iterator:
        while data.endswith(b"&"):
            data += next(iterator)
        yield data.replace(b"&_", b"&amp;_")


def url_str_to_tuple(urls: str | None) -> list[str] | None:
    """
    Convert a string of urls to a list of urls
    """
    if urls is None:
        return None
    else:
        return re.findall(r"https://\S+?(?=(?:https://|,|$))", urls)


class ImageDownloader:
    """
    Class to download images from a list of urls
    """

    def __init__(self):
        self.allowed_formats = {"JPEG", "PNG", "WEBP"}
        self.min_size = (666, 444)

    def load(
        self, id: str | None, image_urls: list[str] | None, output_dir: str
    ) -> None:
        if image_urls is None:
            return
        if id is None:
            return
        with requests.Session() as session:
            for idx, url in enumerate(image_urls):
                try:
                    r = session.get(url, stream=True)
                    if r.status_code != 200:
                        logger.warning(
                            f"Error while loading image from {url}: {r.status_code}"
                        )
                        continue
                    try:
                        img_bytes = b"".join(r.iter_content(None))
                        img = Image.open(io.BytesIO(img_bytes))
                    except (ValueError, TypeError, PIL.UnidentifiedImageError) as e:
                        logger.warning(f"Error while opening image from {url}: {e}")
                        continue

                    # check if image is jpeg, jpg, png or webp
                    if img.format not in self.allowed_formats:
                        logger.warning(f"Invalid image format for {url}: {img.format}")
                        continue

                    # check the size of the image
                    if img.size[0] < self.min_size[0] or img.size[1] < self.min_size[1]:
                        logger.warning(
                            f"Image too small for {url}: {img.size[0]}, {img.size[1]}"
                        )
                        continue

                    try:
                        img.save(
                            f"{output_dir}/{id}_{idx}.{img.format.lower()}",
                            format=img.format,
                        )
                    except ValueError:
                        logger.warning(f"Error while saving image from {url}")

                except requests.exceptions.RequestException as e:
                    logger.warning(f"Error while loading image from {url}: {e}")


@xml_handle_element("yml_catalog", "shop", "offers", "offer")
@dataclass
class RawData:
    """
    Dataclass for parsing xml file
    """

    node: InitVar[Parser]
    id: str | None = field(default=None, init=False)
    category: str | None = field(default=None, init=False)
    group_id: str | None = field(default=None, init=False)
    image_urls: list | None = field(default=None, init=False)
    item_id: str | None = field(default=None, init=False)

    def __post_init__(self, node):
        self.id = node.attributes.get("id")
        self.group_id = node.attributes.get("group_id")

    @xml_handle_element("categories")
    def handle_category(self, node):
        self.category = node.text

    @xml_handle_element("url")
    def handle_url(self, node):
        self.item_id = parse_product_id(node.text)

    @xml_handle_element("picturies")
    def handle_img_links(self, node):
        self.image_urls = url_str_to_tuple(node.text)


def parse(xml_url: str) -> list[RawData]:
    parsed_data = []
    with requests.Session() as session:
        response = session.get(xml_url, stream=True)
        parser = Parser(fix_amp(response.iter_content(None))).iter_from(RawData)

        try:
            for data in parser:
                if None not in (data.id, data.category, data.group_id):
                    parsed_data.append(data)
        except BigXmlError as e:
            logger.warning(f"Error while parsing xml file: {e}")
    return parsed_data


def download_images(
    parsed_data: list[RawData], output_dir: str, num_workers: int = 4
) -> None:
    downloader = ImageDownloader()

    def download_item(item: RawData) -> None:
        downloader.load(item.id, item.image_urls, output_dir)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        _ = list(
            tqdm(
                executor.map(download_item, parsed_data),
                total=len(parsed_data),
            )
        )


def main(
    xml_url: str, output_dir: str, extra_dir: str | None, num_workers: int
) -> None:
    parsed_data = parse(xml_url)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    download_images(parsed_data, output_dir, num_workers)

    good_cod_list = []
    category_list = []
    group_id_list = []
    item_id_list = []

    for data in parsed_data:
        good_cod_list.append(data.id)
        category_list.append(data.category)
        group_id_list.append(data.group_id)
        item_id_list.append(data.item_id)

    df = pd.DataFrame(
        {
            "good_cod": good_cod_list,
            "group_id": group_id_list,
            "category": category_list,
            "item_id": item_id_list,
        }
    )

    p = Path(output_dir).glob("**/*")
    files = [
        [x.resolve().as_posix(), extract_good_cod(x.stem)] for x in p if x.is_file()
    ]
    df_path = pd.DataFrame(files, columns=["path", "good_cod"])  # type: ignore

    if extra_dir is not None:
        p_extra = Path(extra_dir).glob("**/*")
        extra_files = [
            [x.resolve().as_posix(), extract_good_cod(x.stem)]
            for x in p_extra
            if x.is_file()
        ]
        df_path_extra = pd.DataFrame(extra_files, columns=["path", "good_cod"])  # type: ignore
        logger.info(f"Number of extra images: {len(df_path_extra)}")

        df_path = pd.concat([df_path, df_path_extra], ignore_index=True)

    df_path = pd.merge(df_path, df, how="inner", on=["good_cod"])

    logger.info(f"DataFrame df.csv is saved in {Path(output_dir).parent}")
    logger.info(f"Number of images: {len(df)}")

    df_path.to_csv(f"{Path(output_dir).parent}/df.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build vit dataset.")
    parser.add_argument(
        "--xml_url", type=str, required=True, help="Path to the xml file to parse"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save images",
    )
    parser.add_argument(
        "--extra_dir",
        type=str,
        default=None,
        help="Extra directory with additional images",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers to download images",
    )
    args = parser.parse_args()

    main(
        xml_url=args.xml_url,
        output_dir=args.output_dir,
        extra_dir=args.extra_dir,
        num_workers=args.num_workers,
    )
