#importing pandas library
import pandas as pd


def loading_data():

    ## Loading all the data 
    client_df = pd.read_json("data/client_profiles.json")
    feedback_df = pd.read_json("data/feedback.json")
    marketing_df = pd.read_json("data/marketing_assets.json")
    product_df = pd.read_json("data/product_info.json")
    seo_df = pd.read_json("data/seo_keywords.json")


    # Merge marketing content with client profile
    enriched_df = marketing_df.merge(client_df, how='left', left_on='client_name', right_on='name')

    #Merging with product info only if 'product' column exists
    if 'product' in marketing_df.columns:
        enriched_df = enriched_df.merge(
            product_df,
            how='left',
            left_on=['client_id', 'product'],
            right_on=['client_id', 'product_name']
        )
    else:
        enriched_df["features"] = None
        enriched_df["benefits"] = None
        enriched_df["product"] = None  


    # Rename tone_y to just 'tone' (final brand tone)
    enriched_df.rename(columns={'tone_y': 'tone'}, inplace=True)

    # Drop tone_x, name, product_name to clean up
    enriched_df.drop(columns=['tone_x', 'name', 'product_name'], inplace=True)

    return enriched_df


enriched_df = loading_data()

#Saving the result at the location
output_path = "enriched_with_embeddings.pkl" 
enriched_df.to_pickle(output_path) #Storing it as pickle file


