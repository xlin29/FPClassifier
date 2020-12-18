# Crawling code for running on VisibleV8 and organizing the obtained logs
Change the driver's path and visiblev8 executable path in your device.
## Usage
- install VisibleV8 from https://github.com/wspr-ncsu/visiblev8.
- extract\_features.py will automatically process VV8 logs in the ../logs directory.
```bash
python3 crawl.py http://0123putlocker.com/
python3 extract_features.py
```
