from hest import iter_hest

# Iterate through your downloaded sample
for st in iter_hest('../hest_data', id_list=['TENX96']):
    print(f"🔬 Sample: {st.adata}")  # Spatial transcriptomics data
    print(f"🔬 WSI: {st.wsi}")       # Histology image
    print(f"🔬 Metadata: {st.meta}") # Sample metadata
    # save the sample to tenx96/
    # st.save('tenx96')
