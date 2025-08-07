1. The program renders a damselfly using a path tracer that includes Cook Torrance BRDF
2. Run the program in CLion like normal, with the argument set to "damselfly.obj"
3. See the writeup for more technical details on the following: Cook Torrance, Importance Sampling, Volumetric Effects, etc.
4. \begin{itemize}
   \item I will render a flapping damselfly perched on a small branch against a blurry natural backdrop
   \item I will use path tracing to simulate the lighting properties in the objectives below
   \item The bug's body will be modeled in Blender using a fuzzy conductive material, with the light reflection/refraction implemented with Cook
   Torrance BRDF, which adheres to the energy conservation principle and create more accurate renderings of reflections/refractions (compared to Blinn-Phong model for instance)
   \item The bug's wing skeleton will be modeled in Blender and use a dielectric material in the wing's body, with lighting using Fresnel reflection which will control the amount of light that is reflected vs. refracted
   \item There will be an animation of wing flapping based on various physics-based approaches such as a mass-spring system to simulate wing deformation, longitudinal/transverse oscillations for the cycles of wing flapping, and aerodynamic forces that take into consideration the rotational force of flapping wings and direct flight of damselflies
   \item The natural backdrop will include a noise-based terrain for the ground and possibly a particle system for dust/smaller bugs/pollen in the air
   \item Depth of field (e.g. Bokeh) will be implemented to put the bug in focus, while applying different blurs for each layer while rendering more distant objects at a lower resolution, and using motion blur for the flapping wings
   \item Volumetric scattering will be used to emulate the lighting in a forest, such as light rays through the canopy and possible dust floating visibly under direct light