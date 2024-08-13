import flowio
import numpy



fcs_data = flowio.FlowData(r"C:\Users\bruno\OneDrive\Desktop\Programmer\viSNE_maps_and_data\Data\NewProtocolSingleSpecies\iLink_species\Test\Chlorella_vulgaris_1_new.fcs",ignore_offset_error=True)
npy_data = numpy.reshape(fcs_data.events, (-1, fcs_data.channel_count))