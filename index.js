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
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, "./uploads/");

  },  
  filename: (req, file, cb) => {
    cb(null, Date.now() + path.extname(file.originalname)); // Save with a unique name ->
  },
});
const upload = multer({ storage: storage });

// Create an upload folder if it doesn't exist
if (!fs.existsSync("./uploads")) {
  fs.mkdirSync("./uploads");
}
app.use("/uploads", express.static(path.join(__dirname, "uploads")));
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
  if (!req.file) {
    return res.status(400).send("No image uploaded.");
  }

  await loadModels();

  const IMAGE_URL = path.join(__dirname, "uploads", req.file.filename);

  // Check if the image file exists
  if (!fs.existsSync(IMAGE_URL)) {
    return res.status(404).send("Image file not found.");
  }

  // Load the image
  const img = await loadImage(IMAGE_URL);

  if (!img.width || !img.height) {
    return res
      .status(400)
      .send("Failed to load image. Image dimensions are invalid.");
  }

  // Create the canvas and set its dimensions
  const canvas = createCanvas(img.width, img.height);
  const ctx = canvas.getContext("2d");

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(img, 0, 0, img.width, img.height);

  // Perform face detection
  const detections = await faceapi
    .detectAllFaces(canvas)
    .withFaceLandmarks()
    .withFaceDescriptors()
    .withAgeAndGender();

  const output = {
    age: [],
    gender: [],
    ageProbabilities: [],
    genderProbabilities: [],
    imageUrl: "", // This will hold the result image URL
  };

  // Process detections and blur faces
  for (const face of detections) {
    const { top, left, width, height } = face.detection.box;

    // Ensure the values are integers
    const integerLeft = Math.floor(left);
    const integerTop = Math.floor(top);
    const integerWidth = Math.floor(width);
    const integerHeight = Math.floor(height);

    // Crop the face region and apply the blur effect using sharp
    const faceRegion = await sharp(IMAGE_URL)
      .extract({
        left: integerLeft,
        top: integerTop,
        width: integerWidth,
        height: integerHeight,
      }) // Crop the face area
      .blur(7) // Apply blur to the face region
      .toBuffer(); // Convert the face region to a buffer

    // Once the face region is blurred, draw the blurred face back onto the canvas
    const blurredFace = await loadImage(faceRegion); // Ensure the image is fully loaded before drawing
    ctx.drawImage(
      blurredFace,
      integerLeft,
      integerTop,
      integerWidth,
      integerHeight
    );

    // Add text with age and gender
    const { age, gender, genderProbability } = face;
    output.age.push(Math.round(age)); // Add age
    output.gender.push(gender); // Add gender
    output.ageProbabilities.push(Math.round(age * 100) / 100); // Add age probability
    output.genderProbabilities.push(Math.round(genderProbability * 100)); // Add gender probability

    const genderText = `${gender} (${Math.round(genderProbability * 100)}%)`;
    const ageText = `${Math.round(age)} years`;

    // ctx.font = "16px Arial";
    // ctx.fillStyle = "white";
    // ctx.fillText(`${genderText}, ${ageText}`, integerLeft, integerTop - 10);
  }

  // Save the processed image
  const outputPath = path.join(
    __dirname,
    "uploads",
    `result_${req.file.filename}`
  );
  const out = fs.createWriteStream(outputPath);
  const stream = canvas.createPNGStream();
  stream.pipe(out);

  out.on("finish", () => {
    output.imageUrl = `/uploads/result_${req.file.filename}`; // URL of the processed image
    res.status(200).json(output); // Send the data back to the client
  });
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

const PORT = 4000;

app.listen(PORT, () => {
  console.log(
    `Server connected to MongoDB & running on http://localhost:${PORT}`
  );
});
