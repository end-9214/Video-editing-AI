from bing_image_downloader import downloader

# Search and download 10 images of "mountains"
downloader.download(
    query="computer industry",
    limit=1,
    output_dir="images",
    adult_filter_off=True,
    force_replace=False,
    timeout=60
)
