Work iteratively to improve the quality of the visual rendering while keeping
the code manageable.

Ensure that for each kind of rendering (scene, object and texture), the
notebooks generate grids with plt subplots with 5 columns for different seeds
and as many rows there are of rendering types to do.

Make sure that the notebooks are fast to run. Each individual cell should
execute in a maximum of 10 seconds.

Ensure that each individual image stays below 5 MB. Adjust the resolution of the image
to find a good tradeoff between iteration speed, resource usage and enough image quality
to be able to progressively increase realism of the generated images.

At each iteration:

- Execute all the notebooks to make sure that outputs are up to date;
- Extract the images, and below each image, describe, in the notebook, what you
  see in the image;
- Inspect the image to identify things that are good, and list defects that should be
  fixed and assign a severity level to each. Write that after the image descriptions in
  the notebook;
- Consolidate all such visual inspection results in VISUAL_QA.md and then
  re-prioritize next steps based on the current results;
- Clean up the backlog to remove problems that are no longer visible in the
  latest round of visual inspection;
- Proceed with addressing the concerns of the next step with the highest
  priority (one step per iteration);
- Keep a log of what you did in JOURNAL.md file. Use timestamped entries.

When doing the visual analysis of images in notebooks, delete the analysis results of the
previous iteration(s). The write the following for each image:

- description of the current image content;
- list of defects with severity scores;
- assessment of the level of diversity across seeds.

When inspecting images, make sure to never inspect more than a few images at a
time so that the total upload/download size stay will below 30 MB.

Once all major defects are fixed. Step back and think about the how to improve
realism a bit: consistency of view angle, shadows, consistency of the scale of
objects across scenes, consistency of the topology of relationships between
nearby objects, varying density of objects, varying spatial distribution of
objects, lack of diversity accross seeds...

Make sure that the sampled scenes showcase combinations of all the available
vocabulary of objects and textures in realistic enough ways. If some objects or
textures cannot naturally show up in satellite images, consider trimming them
from the library instead.

Also think about progressively enriching the vocabulary of textures, objects
and types of scenes to build more realistic and diverse scenes.

Also consider including cloud cover and other forms of athmospheric / sensor noise.

