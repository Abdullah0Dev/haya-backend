require("dotenv").config();
const tf = require("@tensorflow/tfjs-node");

tf.setBackend("cpu"); // Optional for debugging

// require("@tensorflow/tfjs-node-gpu");
const express = require("express");
const cors = require("cors");
const faceapi = require("face-api.js");
const path = require("path");
const sharp = require("sharp");
const { Canvas, Image, ImageData, loadImage, createCanvas } = require("canvas");
const fs = require("fs");
const multer = require("multer");
const { ClarifaiStub, grpc } = require("clarifai-nodejs-grpc");
const { hostname } = require("os");
const stub = ClarifaiStub.grpc();
const metadata = new grpc.Metadata();
metadata.set("authorization", `Key ${process.env.CLARIFAI_API_KEY}`);

const app = express();
app.use(express.json());
app.use(cors());
// Set up multer for image upload
const storage = multer.memoryStorage(); // Change to memory storage

const upload = multer({ storage: storage });

// Create an upload folder if it doesn't exist
if (!fs.existsSync("./uploads")) {
  fs.mkdirSync("./uploads");
}

faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

app.get("/", (req, res) => {
  res.send("السلام عليكم اهل بك");
});

const loadModels = async () => {
  const MODEL_URL = path.join(__dirname, "models");

  try {
    await Promise.all([
      faceapi.nets.ssdMobilenetv1.loadFromDisk(MODEL_URL),
      faceapi.nets.faceLandmark68Net.loadFromDisk(MODEL_URL),
      faceapi.nets.faceRecognitionNet.loadFromDisk(MODEL_URL),
      faceapi.nets.ageGenderNet.loadFromDisk(MODEL_URL),
    ]);
  } catch (error) {
    console.log("Problem loading models:", error);
  }
};

// Upload image and process it
app.post("/blur-face", upload.single("image"), async (req, res) => {
    try {
      console.log("Uploaded file:", req.file); // Log the file to check if it's populated
  
      if (!req.file) {
        return res.status(400).send("No image uploaded.");
      }
      if (!req.file.buffer) {
        return res.status(400).send("Image buffer is missing.");
      }
  
      const base64Image = req.file.buffer.toString("base64"); // Convert buffer to base64 once
  
      // Use the buffer to make prediction
      const response = await new Promise((resolve, reject) => {
        stub.PostModelOutputs(
          {
            model_id: "aaa03c23b3724a16a56b629203edc62c", // General model ID
            inputs: [
              {
                data: {
                  image: {
                    base64: base64Image,
                  },
                },
              },
            ],
          },
          metadata,
          (err, response) => {
            if (err) {
              reject(err);
              return;
            }
            if (response.status.code !== 10000) {
              reject(response.status.description);
              return;
            }
            resolve(response);
          }
        );
      });
  
      // Extract the description from Clarifai's response
      const description = response.outputs[0].data.concepts[0].name;
  
      await loadModels();
  
      // Directly use the buffer in Sharp
      const img = await sharp(req.file.buffer)
        .metadata();  // Get image metadata (dimensions, etc.)
  
      if (!img.width || !img.height) {
        throw new Error("Failed to load image. Image dimensions are invalid.");
      }
  
      // Create the canvas and set its dimensions
      const canvas = createCanvas(img.width, img.height);
      const ctx = canvas.getContext("2d");
  
      ctx.clearRect(0, 0, canvas.width, canvas.height);
  
      // Create an image from the buffer and draw it to the canvas
      const loadedImage = await loadImage(req.file.buffer); // Load image from buffer
      ctx.drawImage(loadedImage, 0, 0, img.width, img.height);
  
      console.log("Image dimensions:", img.width, img.height);
      console.log("Canvas dimensions:", canvas.width, canvas.height);
  
      if (canvas.width === 0 || canvas.height === 0) {
        throw new Error("Canvas dimensions are invalid.");
      }
  
      try {
        ctx.getImageData(0, 0, img.width, img.height);
      } catch (err) {
        throw new Error("Failed to access canvas image data: " + err.message);
      }
  
      // Perform face detection
      const detections = await faceapi
        .detectAllFaces(canvas)
        .withFaceLandmarks()
        .withFaceDescriptors()
        .withAgeAndGender();
  
      // Process detections and create the output
      const output = {
        age: [],
        gender: [],
        ageProbabilities: [],
        genderProbabilities: [],
        imageUrl: "", // This will hold the result image URL
      };
  
      for (const face of detections) {
        const { top, left, width, height } = face.detection.box;
  
        // Ensure the values are integers
        const integerLeft = Math.floor(left);
        const integerTop = Math.floor(top);
        const integerWidth = Math.floor(width);
        const integerHeight = Math.floor(height);
  
        // Crop and blur the face region using sharp
        const faceRegion = await sharp(req.file.buffer) // Use buffer instead of file path
          .extract({
            left: integerLeft,
            top: integerTop,
            width: integerWidth,
            height: integerHeight,
          })
          .blur(10)
          .toBuffer();
  
        const blurredFace = await loadImage(faceRegion); // Load the blurred face image
        ctx.drawImage(
          blurredFace,
          integerLeft,
          integerTop,
          integerWidth,
          integerHeight
        );
  
        const { age, gender, genderProbability } = face;
        output.age.push(Math.round(age)); // Add age
        output.gender.push(gender); // Add gender
        output.ageProbabilities.push(Math.round(age * 100) / 100); // Add age probability
        output.genderProbabilities.push(Math.round(genderProbability * 100)); // Add gender probability
      }
  
      // Save the output image
      const outputPath = path.join(__dirname, "uploads", `result_${description}.png`);
      const out = fs.createWriteStream(outputPath);
      const stream = canvas.createPNGStream();
      stream.pipe(out);
      out.on("finish", () => {
        // Set the URL for the result image
        output.imageUrl = `/uploads/result_${description}.png`;
        res.status(200).json(output); // Send the output data
      });
  
    } catch (error) {
      console.log("Error:", error);
      res.status(500).send("Server error: " + error.message);
    }
  });
  

async function main() {
  await loadModels();

  const IMAGE_URL = path.join(__dirname, "images", "youngman1.png");

  // Check if the image file exists
  if (!fs.existsSync(IMAGE_URL)) {
    throw new Error("Image file not found at " + IMAGE_URL);
  }

  // Load the image
  const img = await loadImage(IMAGE_URL);

  if (!img.width || !img.height) {
    throw new Error("Failed to load image. Image dimensions are invalid.");
  }

  // Create the canvas and set its dimensions
  const canvas = createCanvas(img.width, img.height);
  const ctx = canvas.getContext("2d");

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(img, 0, 0, img.width, img.height);

  console.log("Image dimensions:", img.width, img.height);
  console.log("Canvas dimensions:", canvas.width, canvas.height);

  if (canvas.width === 0 || canvas.height === 0) {
    throw new Error("Canvas dimensions are invalid.");
  }

  try {
    ctx.getImageData(0, 0, img.width, img.height);
  } catch (err) {
    throw new Error("Failed to access canvas image data: " + err.message);
  }

  // Perform face detection
  const detections = await faceapi
    .detectAllFaces(canvas)
    .withFaceLandmarks()
    .withFaceDescriptors()
    .withAgeAndGender();

  console.log("Detections done alhamdullilah:", detections);
}

main();

const PORT =  4000;

app.listen(PORT, () => {
  console.log(
    `Server connected to MongoDB & running on http://localhost:${PORT}`
  );
});
