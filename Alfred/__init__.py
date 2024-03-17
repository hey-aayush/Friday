import json
import weblinx as wl
import lxml

class WebsiteState():
    def __init__(
        self,
        url:str,
        document: str,
        elements_coordinates:str,
        viewport:dict,
        earliest_messages:dict,
        latest_messages:dict
    ):
        self.url=url
        self.document=document
        self.elements_coordinates=elements_coordinates
        self.viewport=viewport
        self.earliest_messages=earliest_messages
        self.latest_messages=latest_messages
        
    @classmethod
    def from_json_file(self,file_path):
        with open(file_path, "r") as json_file:
            json_data=json.load(json_file)
            return self(
                url=json_data['url'],
                document=json_data['document'],
                elements_coordinates=json_data['elements_coordinates'],
                viewport=json_data['viewport'],
                earliest_messages=json_data['earliest_messages'],
                latest_messages=json_data['latest_messages']
            )

    def filter_element_in_vw(self, min_height=10, min_width=10):
        viewport_height=self.viewport['height']
        viewport_width=self.viewport['width']
        remaining_visible_elements = {}
        for box in self.elements_coordinates:
        # The width and height must be positive
            if box["width"] <= 0 or box["height"] <= 0:
                continue
            if box["width"] < min_width and box["height"] < min_height:
                continue
            if viewport_height is not None and box["y"] > viewport_height:
                continue
            if viewport_width is not None and box["x"] > viewport_width:
                continue
            remaining_visible_elements[box['element_uid']] = box
        return remaining_visible_elements
    
    def build_dmr_input(self,uid_key='data-webtasks-id='):
        remaining_visible_elements=self.filter_element_in_vw()
        root = lxml.html.fromstring(self.document)
        root_tree = root.getroottree()
        elements = root.xpath(f"//*[@{uid_key}]")
        elements_filt = [p for p in elements if p.attrib[uid_key] in remaining_visible_elements]
        
    
WebsiteState.from_json_file("C:\\Users\\ashandilya\\Desktop\\FHL\\Friday\\Alfred\\sampleInput.json")
