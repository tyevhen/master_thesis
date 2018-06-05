import time, os
from urllib import parse
from pyvirtualdisplay import Display
from selenium import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


QUERY = 'my life clinical depression anxiety  anxiety OR depression OR disorder site:blogspot.com'

def init_driver():
    driver_options = webdriver.ChromeOptions()
    # driver_options.add_argument('--proxy-server=202.79.54.7:8080')
    # capa = DesiredCapabilities.CHROME
    # capa["pageLoadStrategy"] = "none"
    driver = webdriver.Chrome(chrome_options=driver_options)
    driver.wait = WebDriverWait(driver, 5)
    return driver

def get_blog_urls(driver, num_pages):
    pretty_urls = []
    domain_names = []
    for i in range(num_pages):
        search_results = driver.find_elements_by_class_name('g')
        with open('ill_urls.txt', 'a', encoding='utf8') as urls_file:
            for res in search_results:
                url = res.find_element_by_css_selector('a').get_attribute('href')
                domain_name = parse.urlparse(url).hostname.split('.')[0]
                if domain_name not in domain_names:
                    domain_names.append(domain_name)
                    pretty_urls.append(url)
                    urls_file.write(url+"\n")
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        driver.find_elements_by_class_name('pn')[-1].click()
        time.sleep(2)
        urls_file.close()
    return pretty_urls

def blog_scraper(driver, url):
    try:
        driver.get(url)
        url = parse.urlparse(url)
        domain_name = url.hostname.split('.')[0]
        print(domain_name)
    except:
        print('Timeout')
    all_urls = scrap_archive(driver)
    if all_urls is not None:
        for i, link in enumerate(all_urls):
            try:
                driver.get(link)
                driver.wait.until(EC.presence_of_element_located((By.CLASS_NAME, 'post-body')))
                driver.execute_script("window.stop();")
                if driver.find_elements_by_class_name('post-body'):
                    posts = driver.find_elements_by_class_name('post-body')
                    write_posts(posts, domain_name+str(i))
                elif driver.find_elements_by_class_name('entry-content'):
                    posts = driver.find_elements_by_class_name('entry-content')
                    write_posts(posts, domain_name + str(i))
                elif driver.find_elements_by_class_name('post-outer'):
                    posts = driver.find_elements_by_class_name('post-outer')
                    write_posts(posts, domain_name + str(i))
            except:
                pass



def scrap_archive(driver):
    try:
        if driver.find_element_by_xpath('//ul[@class="flat"]'):
            flat_list = driver.find_element_by_class_name("flat")
            post_urls = [x.get_attribute('href') for x in flat_list.find_elements_by_xpath("//li[@class='archivedate']/a")]
            return post_urls
            print("Flat archive")
    except:
        print("Flat archive not found")
    # // *[ @ id = "BlogArchive1_ArchiveMenu"] / option[2]
    try:
        if driver.find_element_by_id('BlogArchive1_ArchiveMenu'):
            select_bar = driver.find_element_by_id('BlogArchive1_ArchiveMenu')
            post_urls = [x.get_attribute("value") for x in select_bar.find_elements_by_tag_name("option")]
            return post_urls
            print("Dropdown archive")
    except:
        print("Dropdown archive not found")


    try:
        if driver.find_elements_by_xpath('//ul[@class="hierarchy"]/li[contains(@class, "archivedate") and contains(@class, "collapsed")]'):
            while len(driver.find_elements_by_xpath('//ul[@class="hierarchy"]/li[contains(@class, "archivedate") and contains(@class, "collapsed")]')) != 0:
                toggle = driver.find_elements_by_xpath('//ul[@class="hierarchy"]/li[contains(@class, "archivedate") and contains(@class, "collapsed")]/a[@class="toggle"]')
                for tog in toggle:
                    try:
                        tog = driver.wait.until(EC.element_to_be_clickable((By.XPATH, '//li[contains(@class, "archivedate") and contains(@class, "collapsed")]/a[@class="toggle"]')))
                        tog.click()
                        time.sleep(3)
                    except:
                        print("Can't toggle")
                        pass

            post_urls = [url.get_attribute('href') for url in driver.find_elements_by_xpath('//ul[@class="posts"]/li/a')]
            return post_urls
    except:
        print("Toggle archive not found")

    try:
        if driver.find_element_by_xpath('//div[@id="LinkList1"]'):
            post_urls = [url.get_attribute('href') for url in driver.find_elements_by_xpath('//div[@id="LinkList1"]/div/ul/li/a')]
            return post_urls
            print("Link list")
    except:
        print("Link list not found")


    try:
        if driver.find_element_by_class_name('popular-posts'):
            post_urls = [url.get_attribute('href') for url in driver.find_elements_by_xpath('//div[@class="item-title"]/a')]
            return post_urls
            print('Popular posts')
    except:
        print('Popular posts section not found')


def write_posts(post, filename):
    with open(filename + ".txt", "w") as blog:
        print(post[0].text[:15])
        blog.write(post[0].text)
    blog.close()

def read_urls_file(filename):
    with open(filename, 'r', encoding='utf8') as urls_source:
        urls = [url.strip('\n') for url in  urls_source.readlines()]
    urls_source.close()
    return urls


if __name__ == "__main__":
    driver = init_driver()
    driver.get('https://www.google.com')
    ill_path = 'ill_blogs'
    healthy_path = 'healthy_blogs'
    # try:
    #     input_field = driver.wait.until(EC.presence_of_element_located((By.NAME, 'q')))
    #     button = driver.wait.until(EC.element_to_be_clickable((By.NAME, "btnK")))
    #     time.sleep(5)
    #     for ch in QUERY:
    #         input_field.send_keys(ch)
    #         time.sleep(0.01)
    #     button.click()
    # except TimeoutException:
    #     print('Timeout')

    blog_urls = read_urls_file('healthy_urls.txt')
    if not os.path.exists(healthy_path):
        os.makedirs(healthy_path)
    os.chdir(healthy_path)

    # flat_url_list = [y for x in blog_urls for y in x]
    for url in blog_urls:
        blog_scraper(driver, url)








