collector.py

Downloads images by sequential numeric ID using a URL template.

Usage:

python collector.py --url-template "https://example.com/images/{id}.jpg" --outdir images --start 1 --end 100

Or stop after a number of consecutive misses:

python collector.py --url-template "https://example.com/images/{id}.jpg" --outdir images --start 1 --max-misses 50

Options:
- --url-template: URL with {id} placeholder
- --outdir: folder to save images (created if missing)
- --start, --end: numeric ID range
- --max-misses: stop after N consecutive misses if --end omitted
- --concurrency: number of parallel downloads
- --overwrite: overwrite existing files

Requirements:
- Python 3.8+
- requests
