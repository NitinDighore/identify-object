import React, { useEffect, useRef, useState } from "react";
// ---------------------------------------------------------------
import Select, { SelectChangeEvent } from "@mui/material/Select";
import {
  Grid,
  Box,
  InputLabel,
  MenuItem,
  FormControl,
  Stack,
  Button,
  ImageList,
  ImageListItem,
} from "@mui/material";
// ---------------------------------------------------------------
const tf = require("@tensorflow/tfjs");
const mobilenetModule = require("@tensorflow-models/mobilenet");
const knnClassifier = require("@tensorflow-models/knn-classifier");
// ---------------------------------------------------------------

function App() {
  // Ref
  const exampleImageRef: any = useRef();
  const identifyImageRef: any = useRef();

  // useStates
  const [className, setClassName] = useState("");
  const [images, setImages] = useState<any>([]);
  const [imagesUrls, setImagesUrls] = useState<any>([]);
  const [toIdentifyImage, setToIdentifyImages] = useState<any>([]);
  const [toIdentifyImageUrls, setToIdentifyImagesUrls] = useState<any>([]);
  const [predication, setPredication] = useState<any>(null);
  const [classifier, setClassifier] = useState<any>(null);
  const [mobilenet, setMobilenet] = useState<any>(null);

  // Module Loading
  const loadModules = async () => {
    // Create the classifier.
    const classifierInstance = await knnClassifier.create();
    setClassifier(classifierInstance);

    // Load mobilenet.
    const mobilenetInstance = await mobilenetModule.load();
    setMobilenet(mobilenetInstance);
  };

  useEffect(() => {
    loadModules();
  }, []);

  // ClassName
  const handleChange = (event: SelectChangeEvent) => {
    setClassName(event.target.value as string);
  };

  // Uploading Images for examples
  const uploadImage = (e: any) => {
    const { files } = e.target;

    if (files.length > 0) {
      const imagesArray = Object.values(files);

      setImages(imagesArray);

      const imagesUrls = imagesArray.map((image: any) => {
        const url = URL.createObjectURL(image);
        return { img: url };
      });

      setImagesUrls(imagesUrls);
    }
  };

  const createExamples = async () => {
    console.log(className);

    const imageInPixelForm = tf.browser.fromPixels(exampleImageRef.current);
    const inferredImage = mobilenet.infer(imageInPixelForm, className);
    classifier.addExample(inferredImage, className);
  };

  // Uploading Image for identification
  const uploadImageIdentification = (e: any) => {
    const { files } = e.target;

    if (files.length > 0) {
      const imagesArray = Object.values(files);

      setToIdentifyImages(imagesArray);

      const imagesUrls = imagesArray.map((image: any) => {
        const url = URL.createObjectURL(image);
        return { img: url };
      });

      setToIdentifyImagesUrls(imagesUrls);
    }
  };

  const verifyImage = async () => {
    const imageInPixelForm = tf.browser.fromPixels(identifyImageRef.current);
    const inferredImage = mobilenet.infer(imageInPixelForm, true);
    console.log("Predictions:");
    const classPredication = await classifier.predictClass(inferredImage);
    console.log(classPredication);

    setPredication(classPredication);
  };

  return (
    <Grid className="App" padding={5}>
      <Box sx={{ minWidth: 120 }}>
        <FormControl fullWidth>
          <InputLabel id="demo-simple-select-label">Class Name</InputLabel>
          <Select
            labelId="demo-simple-select-label"
            id="demo-simple-select"
            value={className}
            label="Class Name"
            onChange={handleChange}
          >
            <MenuItem value={"Car"}>Car</MenuItem>
            <MenuItem value={"Bike"}>Bike</MenuItem>
            <MenuItem value={"Fan"}>Fan</MenuItem>
            <MenuItem value={"Mobile"}>Mobile</MenuItem>
          </Select>
        </FormControl>
      </Box>
      <br />
      <Stack direction="row" alignItems="center" spacing={2}>
        <Button variant="contained" component="label">
          Upload
          <input hidden accept="image/*" type="file" onChange={uploadImage} />
        </Button>
      </Stack>
      {imagesUrls.length ? (
        <Grid>
          <ImageList sx={{ width: 500, height: 200 }} cols={3} rowHeight={164}>
            {imagesUrls.map((item: any) => (
              <ImageListItem key={item.img}>
                <img
                  src={`${item.img}`}
                  srcSet={`${item.img}`}
                  // alt={item.title}
                  loading="lazy"
                  ref={exampleImageRef}
                />
              </ImageListItem>
            ))}
          </ImageList>
          <br />
          <Stack direction="row" alignItems="center" spacing={2}>
            <Button
              variant="contained"
              component="label"
              onClick={createExamples}
            >
              Create Example
            </Button>
          </Stack>
        </Grid>
      ) : (
        ""
      )}
      <br />
      <Stack direction="row" alignItems="center" spacing={2}>
        <Button variant="contained" component="label">
          Upload Image To Identify
          <input
            hidden
            accept="image/*"
            type="file"
            onChange={uploadImageIdentification}
          />
        </Button>
      </Stack>
      {toIdentifyImageUrls.length ? (
        <Grid>
          <ImageList sx={{ width: 500, height: 200 }} cols={3} rowHeight={164}>
            {toIdentifyImageUrls.map((item: any) => (
              <ImageListItem key={item.img}>
                <img
                  src={`${item.img}`}
                  srcSet={`${item.img}`}
                  // alt={item.title}
                  loading="lazy"
                  ref={identifyImageRef}
                />
              </ImageListItem>
            ))}
          </ImageList>
          <br />
          <Stack direction="row" alignItems="center" spacing={2}>
            <Button variant="contained" component="label" onClick={verifyImage}>
              Verify Images
            </Button>
          </Stack>
        </Grid>
      ) : (
        ""
      )}
      {JSON.stringify(predication)}
    </Grid>
  );
}

export default App;
