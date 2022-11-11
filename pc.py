from fdasrsf import fPCA, time_warping, fdawarp, fdahpca

# Functional Alignment
# Align time-series
warp_f = time_warping.fdawarp(f, time)
warp_f.srsf_align()

warp_f.plot()

# Functional Principal Components Analysis

# Define the FPCA as a vertical analysis
fPCA_analysis = fPCA.fdavpca(warp_f)

# Run the FPCA on a 3 components basis 
fPCA_analysis.calc_fpca(no=3)
fPCA_analysis.plot()

import plotly.graph_objects as go

# Plot of the 3 functions
fig = go.Figure()

# Add traces
fig.add_trace(go.Scatter(y=fPCA_analysis.f_pca[,0,0], mode='lines', name=PC1))
fig.add_trace(go.Scatter(y=fPCA_analysis.f_pca[,0,1], mode='lines', name=PC2))
fig.add_trace(go.Scatter(y=fPCA_analysis.f_pca[,0,2], mode='lines', name=PC3))

fig.update_layout(
    title_text='bPrincipal Components Analysis Functionsb', title_x=0.5,
)

fig.show()

# Coefficients of PCs against regions
fPCA_coef = fPCA_analysis.coef

# Plot of PCs against regions
fig = go.Figure(data=go.Scatter(x=fPCA_coef[,0], y=fPCA_coef[,1], mode='markers+text', text=df.columns))

fig.update_traces(textposition='top center')

fig.update_layout(
    autosize=False,
    width=800,
    height=700,
    title_text='bFunction Principal Components Analysis on 2018 French Temperaturesb', title_x=0.5,
    xaxis_title=PC1,
    yaxis_title=PC2,
)
fig.show()