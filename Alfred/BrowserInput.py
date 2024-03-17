import json
import uuid
from selenium import webdriver
from selenium.webdriver.common.by import By

def get_element_position(driver, element):
    # Get the location (coordinates) of the element relative to the top-left corner of the viewport
    element_uid = element.get_attribute("data-webtasks-id")
    location = element.location
    x = location['x']
    y = location['y']

    # Get the size (width and height) of the element
    size = element.size
    width = size['width']
    height = size['height']
    left = x
    top = y
    right = left + width
    bottom = top + height

    return{
        "element_uid":element_uid,
        "dimension": {
                "x": x,
                "y": y,
                "width": width,
                "height": height,
                "top": top,
                "bottom": bottom,
                "left": left,
                "right": right
            }
        }

def main(url):
    # Initialize a WebDriver instance (make sure you have the appropriate browser driver installed)
    driver = webdriver.Chrome()

    try:
        # Navigate to the webpage
        driver.get("file:///" + "C:\\Users\\ashandilya\\Desktop\\FHL\Friday\\Alfred\\page-3-0.html")
        url = driver.current_url
        # Add a unique UID tag to each element
        body_element = driver.find_element(By.TAG_NAME, "body")
        # Find all elements within the <body> tag
        body_elements = body_element.find_elements(By.XPATH, ".//*")
        
        # for index, element in enumerate(all_elements):
        #     element_uid = uuid.uuid4().hex
        #     driver.execute_script(f"arguments[0].setAttribute('data-webtasks-id', '{element_uid}')", element)

        # Get information about the viewport
        viewport_width = driver.execute_script("return document.documentElement.clientWidth")
        viewport_height = driver.execute_script("return document.documentElement.clientHeight")

        # Collect position coordinates for each element
        elements_info = []
        for element in body_elements:
            element_uid = element.get_attribute("data-webtasks-id")
            if element_uid:
                elements_info.append(get_element_position(driver, element))

        # Fetch the complete HTML of the page
        html_content = driver.page_source
        
        # Construct the JSON object
        # Include Utterance first last 4 and all
        
        result = {
            "url":url,
            "document": html_content,
            "elements_coordinates": elements_info,
            "viewport": {
                "width": viewport_width,
                "height": viewport_height
            },
            "earliest_messages":[
                {
                    "timestamp": -15.449,
                    "speaker": "instructor",
                    "utterance": "Hello",
                    "type": "chat"
                },
                {
                    "timestamp": -11.449,
                    "speaker": "navigator",
                    "utterance": "Hi",
                    "type": "chat"
                },
                {
                    "timestamp": 3.551,
                    "speaker": "instructor",
                    "utterance": "Please open the Stack Exchange website.",
                    "type": "chat"
                },
                {
                    "timestamp": 9.551,
                    "speaker": "navigator",
                    "utterance": "Sure.",
                    "type": "chat"
                }    
            ],
            "latest_messages":[
                {
                    "timestamp": 41.551,
                    "speaker": "navigator",
                    "utterance": "How can I help you?",
                    "type": "chat"
                },
                {
                    "timestamp": 45.551,
                    "speaker": "instructor",
                    "utterance": "Send me the top 8 questions from stack exchange.",
                    "type": "chat"
                },
                {
                    "timestamp": 51.551,
                    "speaker": "navigator",
                    "utterance": "Alright.",
                    "type": "chat"
                }
            ],
            "latest_actions":[
                
            ]
        }

        with open("sampleInput.json", "w") as outfile:
            outfile.write(json.dumps(result, indent=4))

    finally:
        # Close the browser session
        driver.quit()

if __name__ == "__main__":
    main("https://example.com")