#include "cs488.h"
CS488Window CS488;

 
// draw something in each frame
static void draw() {
    for (int j = 0; j < globalHeight; j++) {
        for (int i = 0; i < globalWidth; i++) {
            // FrameBuffer.pixel(i, j) = float3(PCG32::rand()); // noise
             FrameBuffer.pixel(i, j) = float3(0.5f * (sqrt((cos((i + globalFrameCount) * 0.1f) + 1.0f)) + sqrt(sin(i + globalFrameCount)))); // moving cosine
        }
    }
}
static void A0(int argc, const char* argv[]) {
    // set the function to be called in the main loop
    CS488.process = draw;
}



// setting up lighting
static PointLightSource light;
static void setupLightSource() {
    //light.position = float3(0.5f, 4.0f, 1.0f); // use this for sponza.obj
    light.position = float3(3.0f, 3.0f, 2.0f);
    light.wattage = float3(1000.0f, 1000.0f, 1000.0f);
    globalScene.addLight(&light);

}



// ======== you probably don't need to modify below in A1 to A3 ========
// loading .obj file from the command line arguments
static TriangleMesh mesh;
static void setupScene(int argc, const char* argv[]) {
    if (argc > 1) {
        bool objLoadSucceed = mesh.load(argv[1]);
        if (!objLoadSucceed) {
            printf("Invalid .obj file.\n");
            printf("Making a single triangle instead.\n");
            mesh.createSingleTriangle();
        }
    } else {
        printf("Specify .obj file in the command line arguments. Example: CS488.exe cornellbox.obj\n");
        printf("Making a single triangle instead.\n");
        mesh.createSingleTriangle();
    }
    globalScene.addObject(&mesh);
    VolumetricEffect canopyVolume (1.0f, float3(0.5f, 0.7f, 0.3f), float3(0.1f, 0.1f, 0.1f), 0.3f);
    globalScene.volumes.push_back(&canopyVolume);
    VolumetricEffect canopyVolume2 (1.0f, float3(2.0f, 2.0f, 0.3f), float3(3.0f, 0.1f, 0.1f), -0.5f);
    globalScene.volumes.push_back(&canopyVolume);

}

/*static TriangleMesh damselfly;
static AnimatedWings wings;

static void setupScene(int argc, const char* argv[]) {
    // Load damselfly body
    if (!damselfly.load("damselfly.obj")) {
        printf("Failed to load damselfly.obj\n");
        damselfly.createSingleTriangle();
    }
    globalScene.addObject(&damselfly);

    // Load wing animation frames
    std::string wingBase = (argc > 2) ? argv[2] : "wings";  // Default to "wings" if not specified
    int numFrames = (argc > 3) ? std::stoi(argv[3]) : 15;   // Default to 15 frames if not specified

    wings.loadAnimation(wingBase, numFrames);
    globalScene.addObject(&wings.getCurrentFrame());
    globalScene.setWings(&wings);
}*/



static void A1(int argc, const char* argv[]) {
    setupScene(argc, argv);
    setupLightSource();
    // globalRenderType = RENDER_RAYTRACE;
    globalRenderType = RENDER_PATHTRACE;
}

static void A2(int argc, const char* argv[]) {
    setupScene(argc, argv);
    setupLightSource();
    globalRenderType = RENDER_RASTERIZE;
}

// ======== you probably don't need to modify above in A1 to A3 ========


int main(int argc, const char* argv[]) {
    //A0(argc, argv);
     A1(argc, argv);
    // A2(argc, argv);
    // A3(argc, argv);

    CS488.start();
}