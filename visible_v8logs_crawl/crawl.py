import json
from selenium import webdriver
from tld import get_tld
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
import os
import sys
import time


BINARY_PATH = '/home/xlin/VisibleV8/visible_chromium/src/out/Builder/chrome'
EXECUTABLE_PATH = '/home/xlin/Tools/chromedriver_75/chromedriver'
TIMESTR = str(time.strftime("%Y%m%d_%H%M%S"))



def run_onedomain(domain_url):
    """
    Run the domain and organize the visiblev8 logs

    param: domain_url: str
    """
    domain_dir = domain_url.replace('/', '_')
    domain_dir = domain_dir.replace(':', '_')
    if not os.path.isdir(domain_dir):
        os.mkdir(domain_dir)
    os.chdir(domain_dir)
    
    caps = DesiredCapabilities().CHROME
    caps["pageLoadStrategy"] = 'normal'  # complete
    options = Options()
    options.headless = False
    options.binary_location = BINARY_PATH
    options.add_argument('--no-sandbox')
    
    try:
        fld = get_tld(domain_url, as_object=True).fld
        driver = webdriver.Chrome(desired_capabilities=caps, executable_path=EXECUTABLE_PATH, chrome_options=options)
        if 'http' not in domain_url:
            domain_url = 'http://'+domain_url
        driver.get(domain_url)
        time.sleep(10)
        print(fld)
    except Exception as e:
        print(str(e))
        exit(1)


    for root, dirs, files in os.walk("."):
        i = 0
        for filename in files:
            if filename.startswith('vv8-') and filename.endswith('0.log'):
                os.rename(filename, fld + '_' + timestr + '_' +str(i) + '.log')
                i += 1
    try:
        driver.quit()
    except Exception as exc:
        print(str(exc))
        exit(1)


if __name__ == '__main__':
    top_url = sys.argv[1]
    run_onedomain(top_url)
