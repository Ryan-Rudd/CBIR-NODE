// CBIR.js

// ---------------------------
// 1. Importing Required Libraries
// ---------------------------
const tf = require('@tensorflow/tfjs'); // TensorFlow.js (Pure JavaScript version)
const mobilenet = require('@tensorflow-models/mobilenet'); // Pre-trained MobileNet model
const sharp = require('sharp'); // Image processing library
const fs = require('fs'); // File system module
const path = require('path'); // Path utilities
const express = require('express'); // Express.js for API
const bodyParser = require('body-parser'); // Middleware for parsing request bodies

// ---------------------------
// 2. Configuration
// ---------------------------
const DATASET_PATH = './dataset'; // Path to the dataset directory
const FEATURES_FILE = './features.json'; // Path to store feature vectors
const IMAGE_SIZE = 224; // MobileNet's expected input size
const TOP_K = 5; // Number of top matches to return

// ---------------------------
// 3. Helper Function: Load and Preprocess Image
// ---------------------------
/**
 * Loads an image from the given path, resizes it, and preprocesses it for MobileNet.
 * @param {string} imagePath - Path to the image file.
 * @returns {tf.Tensor} - Preprocessed image tensor.
 */
const loadImage = async (imagePath) => {
    try {
        // Read and resize the image using sharp
        const buffer = await sharp(imagePath)
            .resize(IMAGE_SIZE, IMAGE_SIZE) // Resize to 224x224
            .removeAlpha() // Remove alpha channel if present
            .toFormat('png') // Ensure consistent format
            .raw({ channels: 3 }) // Force output to have exactly 3 channels (RGB)
            .toBuffer();

        // Verify buffer length
        const expectedBufferLength = IMAGE_SIZE * IMAGE_SIZE * 3; // 224 * 224 * 3 = 150528
        if (buffer.length !== expectedBufferLength) {
            throw new Error(`Unexpected buffer length for image ${imagePath}: Expected ${expectedBufferLength}, got ${buffer.length}`);
        }

        // Convert raw buffer to a Uint8Array
        const uint8Array = new Uint8Array(buffer);

        // Create a tensor from the raw pixel data
        const tensor = tf.tensor3d(uint8Array, [IMAGE_SIZE, IMAGE_SIZE, 3], 'int32')
            .expandDims(0) // Add batch dimension
            .toFloat()
            .div(255); // Normalize to [0,1]

        return tensor;
    } catch (error) {
        console.error(`Error loading image ${imagePath}:`, error);
        throw error;
    }
};

// ---------------------------
// 4. Helper Function: Extract Features Using MobileNet
// ---------------------------
/**
 * Extracts feature vector from an image using MobileNet.
 * @param {string} imagePath - Path to the image file.
 * @param {object} model - Loaded MobileNet model.
 * @returns {number[]} - Extracted feature vector as an array.
 */
const extractFeatures = async (imagePath, model) => {
    const imageTensor = await loadImage(imagePath);
    const predictions = await model.infer(imageTensor, true); // Get intermediate layer (feature vector)
    const featureArray = predictions.arraySync(); // Convert tensor to array
    return featureArray[0]; // Remove batch dimension
};

// ---------------------------
// 5. Function: Process Dataset and Extract Features
// ---------------------------
/**
 * Processes all images in the dataset directory, extracts their features,
 * and saves them to a JSON file.
 */
const processDataset = async () => {
    try {
        console.log('Loading MobileNet model...');
        const model = await mobilenet.load({ version: 2, alpha: 1.0 }); // Load pre-trained MobileNet (V2)

        const featureIndex = {}; // Object to store image features

        const images = fs.readdirSync(DATASET_PATH); // Read all files in dataset
        console.log(`Found ${images.length} files in the dataset.`);

        for (const image of images) {
            const imagePath = path.join(DATASET_PATH, image);

            // Check if the file is an image (optional)
            const ext = path.extname(image).toLowerCase();
            if (!['.jpg', '.jpeg', '.png', '.bmp', '.gif'].includes(ext)) {
                console.warn(`Skipping non-image file: ${image}`);
                continue;
            }

            console.log(`Processing ${image}...`);
            const features = await extractFeatures(imagePath, model);
            featureIndex[image] = features;
        }

        // Save the feature index to a JSON file
        fs.writeFileSync(FEATURES_FILE, JSON.stringify(featureIndex));
        console.log(`Feature extraction complete. Features saved to ${FEATURES_FILE}.`);
    } catch (error) {
        console.error('Error processing dataset:', error);
    }
};

// ---------------------------
// 6. Helper Function: Compute Cosine Similarity
// ---------------------------
/**
 * Computes the cosine similarity between two vectors.
 * @param {number[]} vecA - First vector.
 * @param {number[]} vecB - Second vector.
 * @returns {number} - Cosine similarity score.
 */
const cosineSimilarity = (vecA, vecB) => {
    const dotProduct = vecA.reduce((sum, a, i) => sum + a * vecB[i], 0);
    const magnitudeA = Math.sqrt(vecA.reduce((sum, a) => sum + a * a, 0));
    const magnitudeB = Math.sqrt(vecB.reduce((sum, b) => sum + b * b, 0));
    return dotProduct / (magnitudeA * magnitudeB);
};

// ---------------------------
// 7. Function: Search for Similar Images
// ---------------------------
/**
 * Searches the dataset for images similar to the query image.
 * @param {string} queryImagePath - Path to the query image.
 * @param {number} topK - Number of top matches to return.
 * @returns {Array} - Array of top matching images with similarity scores.
 */
const searchImage = async (queryImagePath, topK = TOP_K) => {
    try {
        console.log('Loading MobileNet model for searching...');
        const model = await mobilenet.load({ version: 2, alpha: 1.0 }); // Load MobileNet (V2)

        console.log('Extracting features from query image...');
        const queryFeatures = await extractFeatures(queryImagePath, model);

        console.log('Loading feature index...');
        const featureIndex = JSON.parse(fs.readFileSync(FEATURES_FILE, 'utf-8'));

        const results = []; // Array to store similarity results

        console.log('Computing similarities...');
        for (const [image, features] of Object.entries(featureIndex)) {
            const similarity = cosineSimilarity(queryFeatures, features);
            results.push({ image, similarity });
        }

        // Sort results by similarity in descending order
        results.sort((a, b) => b.similarity - a.similarity);

        // Extract the most similar image
        const topMatch = results[0];
        const otherMatches = results.slice(1, topK);

        // Log the most similar image separately
        console.log('\n============================');
        console.log('Most Similar Image:');
        console.log(`Image: ${topMatch.image}`);
        console.log(`Similarity Score: ${topMatch.similarity.toFixed(4)}`);
        console.log('============================\n');

        // Log the remaining top matches
        if (otherMatches.length > 0) {
            console.log(`Top ${TOP_K - 1} Other Matches:`);
            otherMatches.forEach((match, index) => {
                console.log(`${index + 1}. Image: ${match.image} | Similarity Score: ${match.similarity.toFixed(4)}`);
            });
        }

        return results.slice(0, topK);
    } catch (error) {
        console.error('Error searching image:', error);
        throw error;
    }
};

// ---------------------------
// 8. Function: Initialize and Run CBIR
// ---------------------------
/**
 * Main function to process dataset and perform a sample search.
 * Uncomment the desired function calls based on your needs.
 */
const main = async () => {
    // Step 1: Process Dataset (Uncomment if running for the first time or after dataset changes)
    // await processDataset();

    // Step 2: Search for a Query Image
    // Replace './query.jpg' with your query image path
    // const queryImagePath = './query.jpg';
    // await searchImage(queryImagePath);

    // Note: When using the API, you don't need to call these functions here
};

// ---------------------------
// 9. Optional: Express.js API for CBIR
// ---------------------------
/**
 * Sets up an Express.js server with an endpoint to handle image search queries.
 */
const setupAPI = () => {
    const app = express();
    const PORT = 3000;

    // Middleware to parse JSON bodies
    app.use(bodyParser.json());

    /**
     * POST /search
     * Body Parameters:
     * - imagePath: string (path to the query image)
     * 
     * Response:
     * - Object containing the most similar image and top matches with similarity scores
     */
    app.post('/search', async (req, res) => {
        try {
            const { imagePath } = req.body;

            if (!imagePath) {
                return res.status(400).json({ error: 'imagePath is required in the request body.' });
            }

            // Check if the query image exists
            if (!fs.existsSync(imagePath)) {
                return res.status(400).json({ error: `Query image not found at path: ${imagePath}` });
            }

            // Perform the search
            const topMatches = await searchImage(imagePath);

            // Structure the response to highlight the most similar image
            const response = {
                mostSimilar: topMatches[0],
                otherMatches: topMatches.slice(1)
            };

            // Respond with the results
            return res.json(response);
        } catch (error) {
            console.error('API Error:', error);
            return res.status(500).json({ error: 'An error occurred while processing the request.' });
        }
    });

    // Start the server
    app.listen(PORT, () => {
        console.log(`CBIR API is running on http://localhost:${PORT}`);
    });
};

// ---------------------------
// 10. Execute Main or API Based on Arguments
// ---------------------------
/**
 * Allows running the script in two modes:
 * - CLI Mode: Process dataset and perform search.
 * - API Mode: Start the Express.js server.
 * 
 * Usage:
 * - To process dataset: node CBIR.js process
 * - To search image: node CBIR.js search <queryImagePath>
 * - To start API: node CBIR.js api
 */
const args = process.argv.slice(2);

if (args.length === 0) {
    console.log('No arguments provided. Running main function.');
    main();
} else {
    const command = args[0].toLowerCase();

    switch (command) {
        case 'process':
            // Process the dataset to extract and save features
            processDataset();
            break;
        case 'search':
            // Perform a search with the provided query image path
            if (args.length < 2) {
                console.error('Please provide the query image path.');
                console.error('Usage: node CBIR.js search <queryImagePath>');
                process.exit(1);
            }
            const queryImagePath = args[1];
            searchImage(queryImagePath);
            break;
        case 'api':
            // Start the Express.js API server
            setupAPI();
            break;
        default:
            console.error('Unknown command.');
            console.error('Available commands: process, search, api');
            break;
    }
}
