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
  const [isModuleLoading, setIsModuleLoading] = useState(false);
  const [className, setClassName] = useState("");
  const [selectedExampleImage, setSelectedExampleImage] = useState<any>(null);
  const [exampleImagesUrls, setExampleImagesUrls] = useState<any>([]);
  const [selectedIdentifyImage, setSelectedIdentifyImage] = useState<any>(null);
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

    setIsModuleLoading(false);
  };

  useEffect(() => {
    setIsModuleLoading(true);
    loadModules();
  }, []);

  // ClassName
  const handleChange = (event: SelectChangeEvent) => {
    setClassName(event.target.value as string);
  };

  // Uploading Images for examples
  const uploadImage = (e: any) => {
    const { files } = e.target;

    if (files) {
      const imageObject = files[0];

      const exampleImageUrl = URL.createObjectURL(imageObject);

      setSelectedExampleImage({
        img: exampleImageUrl,
        class: className,
      });
    }
  };

  const createExamples = async () => {
    const imageInPixelForm = tf.browser.fromPixels(exampleImageRef.current);
    const inferredImage = mobilenet.infer(imageInPixelForm, className);
    classifier.addExample(inferredImage, className);

    setExampleImagesUrls([
      ...exampleImagesUrls,
      {
        ...selectedExampleImage,
      },
    ]);

    setSelectedExampleImage(null);
  };

  // Uploading Image for identification
  const uploadImageIdentification = (e: any) => {
    const { files } = e.target;

    if (files) {
      const imagesObject = files[0];

      const toIdentifyImageUrl = URL.createObjectURL(imagesObject);

      setSelectedIdentifyImage({
        img: toIdentifyImageUrl,
        class: className,
      });
    }
  };

  const verifyImage = async () => {
    setPredication(null);
    const imageInPixelForm = tf.browser.fromPixels(identifyImageRef.current);
    const inferredImage = mobilenet.infer(imageInPixelForm, true);
    const classPredication = await classifier.predictClass(inferredImage);

    const confidenceValuesArray: any = Object.values(
      classPredication.confidences
    );

    let maximum = confidenceValuesArray[0];
    let maximumIndex = 0;

    for (let i = 0; i < confidenceValuesArray.length; i++) {
      const element = confidenceValuesArray[i];

      if (element > maximum) {
        maximum = element;
        maximumIndex = i;
      }
    }

    const confidenceKeysArray: any = Object.keys(classPredication.confidences);

    const predictionObject = {
      className: confidenceKeysArray[maximumIndex],
      confidence: maximum,
    };

    setPredication(predictionObject);
  };

  return (
    <>
      {isModuleLoading ? (
        "Module Is Loading"
      ) : (
        <Grid container className="App" padding={5}>
          <Grid item xs={12}>
            <Box sx={{ minWidth: 120 }}>
              <FormControl fullWidth>
                <InputLabel id="demo-simple-select-label">
                  Class Name
                </InputLabel>
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
          </Grid>
          <Grid item xs={6}>
            <Grid container>
              <Grid item xs={6}>
                <Stack direction="row" alignItems="center" spacing={2}>
                  <Button variant="contained" component="label">
                    Upload
                    <input
                      hidden
                      accept="image/*"
                      type="file"
                      onChange={uploadImage}
                    />
                  </Button>
                </Stack>
              </Grid>

              <Grid item xs={6}>
                {selectedExampleImage && (
                  <>
                    <Stack direction="row" alignItems="center" spacing={2}>
                      <Button
                        variant="contained"
                        component="label"
                        onClick={createExamples}
                      >
                        Create Example
                      </Button>
                    </Stack>
                  </>
                )}
              </Grid>

              <Grid item xs={6}>
                {selectedExampleImage && (
                  <>
                    <Stack direction="row" alignItems="center" spacing={2}>
                      <img
                        style={{ width: 200, height: 200 }}
                        src={`${selectedExampleImage.img}`}
                        srcSet={`${selectedExampleImage.img}`}
                        alt={"alt"}
                        loading="lazy"
                        ref={exampleImageRef}
                      />
                    </Stack>
                  </>
                )}
              </Grid>
            </Grid>
            <Grid item xs={6}>
              {exampleImagesUrls.length ? (
                <Grid>
                  <ImageList
                    sx={{ width: "200%", height: "200%" }}
                    cols={3}
                    rowHeight={164}
                  >
                    {exampleImagesUrls
                      .filter((image: any) => image.class === className)
                      .map((item: any) => (
                        <ImageListItem key={item.img}>
                          <img
                            src={`${item.img}`}
                            srcSet={`${item.img}`}
                            alt={"alt"}
                            loading="lazy"
                            ref={exampleImageRef}
                          />
                        </ImageListItem>
                      ))}
                  </ImageList>
                </Grid>
              ) : (
                ""
              )}
            </Grid>
          </Grid>
          <Grid item xs={6}>
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

            {selectedIdentifyImage ? (
              <Grid>
                <img
                  style={{ width: 200, height: 200 }}
                  src={`${selectedIdentifyImage.img}`}
                  srcSet={`${selectedIdentifyImage.img}`}
                  alt={"alt"}
                  loading="lazy"
                  ref={identifyImageRef}
                />

                <br />
                <Stack direction="row" alignItems="center" spacing={2}>
                  <Button
                    variant="contained"
                    component="label"
                    onClick={verifyImage}
                  >
                    Verify Images
                  </Button>
                </Stack>

                {predication && (
                  <Grid>
                    ClassName: {predication.className}
                    <br />
                    Confidence: {predication.confidence}
                  </Grid>
                )}
              </Grid>
            ) : (
              ""
            )}
          </Grid>
        </Grid>
      )}
    </>
  );
}

export default App;
