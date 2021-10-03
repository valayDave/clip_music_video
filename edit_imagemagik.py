import xml.etree.ElementTree as ET
IMAGE_MAGIK_XML_PATH = '/etc/ImageMagick-6/policy.xml'
tree = ET.parse(IMAGE_MAGIK_XML_PATH)
root = tree.getroot()
for child in root.getchildren():
    if child.attrib['domain'] == 'path':
        child.set("rights","read,write")
tree.write(IMAGE_MAGIK_XML_PATH)