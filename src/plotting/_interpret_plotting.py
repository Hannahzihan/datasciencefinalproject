import pandas as pd
import matplotlib.pyplot as plt
def coefficient_plotting(fitted_pipeline, preprocessor, numericals, categoricals, sort="No"):
    fitted_preprocessor = fitted_pipeline.named_steps['preprocess']
    glm_model = fitted_pipeline.named_steps['estimate']
    glm_coefficients = glm_model.coef_ 

    spline_transformer = fitted_preprocessor.named_transformers_['numeric'].named_steps['spline']
    spline_feature_names = spline_transformer.get_feature_names_out(numericals)

    ohe_transformer = fitted_preprocessor.named_transformers_['cat']
    ohe_feature_names = ohe_transformer.get_feature_names_out(categoricals)

    all_feature_names = list(spline_feature_names) + list(ohe_feature_names)

    coefficients_df = pd.DataFrame({
        'Feature': all_feature_names,
        'Coefficient': glm_coefficients
    })

    coefficients_df_sorted = coefficients_df.sort_values(by='Coefficient', ascending=False)
    plt.figure(figsize=(10, 8))
    if sort=="No":
       coefficients_df.plot(kind='barh', x='Feature', y='Coefficient', figsize=(10, 20))    
    else:  
        coefficients_df_sorted.plot(kind='barh', x='Feature', y='Coefficient', figsize=(10, 20))   
    plt.title('GLM Coefficients')
    plt.xlabel('Coefficient Value')
    plt.ylabel('Feature')
    plt.show()
