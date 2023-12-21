import os

path = r"C:\Users\bruno\OneDrive\Desktop\Programmer\viSNE_maps_and_data\Data\NewProtocolSingleSpecies\iLink_species\Länder\DK"

list_file=[]
for file in os.listdir(path=path):
    print(file)
    list_file.append(file)

