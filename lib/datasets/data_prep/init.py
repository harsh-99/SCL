from xml.dom import minidom
import xml.etree.cElementTree as ET

category_label = {1: 'car', 2: 'speed limit sign', 3: 'person'}

txt_file = open('test.txt', 'r')
text = txt_file.readlines()

i = 0

x = 0

for i in range(len(text)):
	if text[i][0] == '#':
		path = text[i+1]
		size = text[i+2].split(' ')

		num_obj = int(text[i+4])

		object_list = []

		for j in range(num_obj):
			ob = text[i+j+5].replace('\n', '').split(' ')
			for k in range(len(ob)):
				ob[k] = round(float(ob[k]))
			object_list.append(ob)


		# size = (3, 1208, 1920, 1)
		# object_list = [(1, 1168.4, 627.2, 1247.7, 710.3), (1, 1284.4, 640.8, 1327.5, 681.7)]

		root = ET.Element("annotation")

		folder = ET.SubElement(root, "folder")
		folder.text = 'VOC2007'

		filename = ET.SubElement(root, "filename")
		filename.text = str(x)+'.jpg'

		segment = ET.SubElement(root, "segment")
		segment.text = '0'

		##########

		size_node = ET.SubElement(root, "size")

		width = ET.SubElement(size_node, "width")
		# field1.set("name", "blah")
		width.text = str(size[2])

		height = ET.SubElement(size_node, "height")
		height.text = str(size[1])

		depth = ET.SubElement(size_node, "depth")
		depth.text = str(size[0])

		##########

		# index
		# img path
		# img size: channel height width num
		# 0
		# number of bounding boxes
		# category_label1 x1 y1 x2 y2 (For categories: 1-Car; 2-Speed Limit Sign; 3-Person)
		# category_label2 x1 y1 x2 y2
		# category_label3 x1 y1 x2 y2
		# category_label4 x1 y1 x2 y2

		for object_value in object_list:
			obj = ET.SubElement(root, "object")

			name = ET.SubElement(obj, "name")
			name.text = category_label[object_value[0]]

			# pose = ET.SubElement(obj, "pose")
			# pose.text = "Unspecified"

			truncated = ET.SubElement(obj, "truncated")
			truncated.text = "0"

			difficult = ET.SubElement(obj, "difficult")
			difficult.text = "0"

			bndbox = ET.SubElement(obj, "bndbox")

			xmin = ET.SubElement(bndbox, "xmin")
			xmin.text = str(object_value[1])

			ymin = ET.SubElement(bndbox, "ymin")
			ymin.text = str(object_value[2])

			xmax = ET.SubElement(bndbox, "xmax")
			xmax.text = str(object_value[3])

			ymax = ET.SubElement(bndbox, "ymax")
			ymax.text = str(object_value[4])

		dom = minidom.parseString(ET.tostring(root)).toprettyxml(indent='\t')

		img = Image.open(path)
		img.write('JPEGImages/'+str(x)+'.jpg')

		with open("Annotations/{}.xml".format(str(x)), "w") as f:
			f.write(dom)
		x += 1