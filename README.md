The damselfly has no wings and is thus in distress!


Implemented a path tracer that includes Cook Torrance BRDF, specular GGX and diffuse cosine-weighted importance sampling, and implemented volumetric rendering though the latter is less noticeable in the output. Below is a screenshot of the output with the sample count set to 30 (which is low so the body is somewhat noisy) and resolution 512 x 384.


<img width="509" height="379" alt="image" src="https://github.com/user-attachments/assets/a9abc15a-6598-423d-8e10-9fa1ca8f328d" />


Things I'm still working on:
- flapping wings animation
- more noticeable volumetric effects

