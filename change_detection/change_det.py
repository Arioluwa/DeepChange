# this need revision, I later used otb compute change detection cli command.
#check cmd.sh

# import rasterio
# import pandas as pd
# import seaborn as sns
# import numpy as np
# import matplotlib.pyplot as plt
# # from sklearn.metrics import confusion_matrix

# source_map = "../../../data/theiaL2A_zip_img/output/2018/2018_SITS_data.tif"
# target_map = "../../../data/theiaL2A_zip_img/output/2019/2019_SITS_data.tif"

# source_array = rasterio.open(source_map).read(1)
# target_array = rasterio.open(target_map).read(1)

# source_array = source_array.astype(np.uint8).flatten()
# target_array = target_array.astype(np.uint8).flatten()

# label = ["Dense built-up area", "Diffuse built-up area", "Industrial and commercial areas", "Roads", "Oilseeds (Rapeseed)", "Straw cereals (Wheat, Triticale, Barley)", "Protein crops (Beans / Peas)", "Soy", "Sunflower", "Corn",  "Tubers/roots", "Grasslands", "Orchards and fruit growing", "Vineyards", "Hardwood forest", "Softwood forest", "Natural grasslands and pastures", "Woody moorlands", "Water"]

# df = pd.DataFrame({"source": source_array, "target": target_array})
# confusion_matrix = pd.crosstab(df["source"], df["target"], rownames=['source'], colnames=['target'])

# plt.figure(figsize=(30,15))

# ax = sns.heatmap(confusion_matrix, annot=True, cbar=False, fmt='d', cmap='Blues', xticklabels=label, yticklabels=label)
# ax.set_xticklabels(ax.get_xticklabels(),rotation = 90)

# plt.title()
# plt.show()