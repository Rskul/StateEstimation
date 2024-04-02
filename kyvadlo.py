#Importing numpy and xml parser
import numpy as np
import xml.etree.ElementTree as ET

#Importing data into code
#DV - dataset with Different variance
#SV - dataset with Constatn variance
#OUT - dataset containing outliers
treeDV = ET.parse('measurements_nooutlier_differentvar.xml')
DataDV = treeDV.getroot()
treeSV = ET.parse('measurements_nooutlier_differentvar.xml')
DataSV = treeSV.getroot()
treeOUT = ET.parse('measurements_nooutlier_differentvar.xml')
DataOUT = treeOUT.getroot()

a = float(DataSV[1][0].text)
print(a)
