#include <assert.h>
#include <math.h>
#include <FL/Fl.H>
#include <FL/Fl_Image.H>
#include "features.h"
#include "ImageLib/FileIO.h"

#define PI 3.14159265358979323846


// Compute features of an image.
bool computeFeatures(CFloatImage &image, FeatureSet &features, int featureType, int descriptorType) {
    // TODO: Instead of calling dummyComputeFeatures, implement
    // a Harris corner detector along with a MOPS descriptor.  
    // This step fills in "features" with information necessary 
    // for descriptor computation.

    switch (featureType) {
    case 1:
        dummyComputeFeatures(image, features);
        break;
    case 2:
        ComputeHarrisFeatures(image, features);
        break;
    default:
        return false;
    }

    // TODO: You will implement two descriptors for this project
    // (see webpage).  This step fills in "features" with
    // descriptors.  The third "custom" descriptor is extra credit.
    switch (descriptorType) {
    case 1:
        ComputeSimpleDescriptors(image, features);
        break;
    case 2:
        ComputeMOPSDescriptors(image, features);
        break;
    case 3:
        ComputeCustomDescriptors(image, features);
        break;
    default:
        return false;
    }

    // This is just to make sure the IDs are assigned in order, because
    // the ID gets used to index into the feature array.
    for (unsigned int i=0; i<features.size(); i++) {
        features[i].id = i;
    }

    return true;
}

// Perform a query on the database.  This simply runs matchFeatures on
// each image in the database, and returns the feature set of the best
// matching image.
bool performQuery(const FeatureSet &f, const ImageDatabase &db, int &bestIndex, vector<FeatureMatch> &bestMatches, double &bestDistance, int matchType) {
    vector<FeatureMatch> tempMatches;

    for (unsigned int i=0; i<db.size(); i++) {
        if (!matchFeatures(f, db[i].features, tempMatches, matchType)) {
            return false;
        }

        bestIndex = i;
        bestMatches = tempMatches;
    }

    return true;
}

// Match one feature set with another.
bool matchFeatures(const FeatureSet &f1, const FeatureSet &f2, vector<FeatureMatch> &matches, int matchType) {

    // TODO: We have provided you the SSD matching function; you must write your own
    // feature matching function using the ratio test.

    printf("\nMatching features.......\n");

    switch (matchType) {
    case 1:
        ssdMatchFeatures(f1, f2, matches);
        return true;
    case 2:
        ratioMatchFeatures(f1, f2, matches);
        return true;
    default:
        return false;
    }
}

// Compute silly example features.  This doesn't do anything
// meaningful, but may be useful to use as an example.
void dummyComputeFeatures(CFloatImage &image, FeatureSet &features) {
    CShape sh = image.Shape();
    Feature f;

    for (int y=0; y<sh.height; y++) {
        for (int x=0; x<sh.width; x++) {
            double r = image.Pixel(x,y,0);
            double g = image.Pixel(x,y,1);
            double b = image.Pixel(x,y,2);

            if ((int)(255*(r+g+b)+0.5) % 100 == 1) {
                // If the pixel satisfies this meaningless criterion,
                // make it a feature.

                f.type = 1;
                f.id += 1;
                f.x = x;
                f.y = y;

                f.data.resize(1);
                f.data[0] = r + g + b;

                features.push_back(f);
            }
        }
    }
}

void ComputeHarrisFeatures(CFloatImage &image, FeatureSet &features)
{
    //Create grayscale image used for Harris detection
    CFloatImage grayImage = ConvertToGray(image);

    //Create image to store Harris values
    CFloatImage harrisImage(image.Shape().width,image.Shape().height,1);

    //Create image to store local maximum harris values as 1, other pixels 0
    CByteImage harrisMaxImage(image.Shape().width,image.Shape().height,1);

    CFloatImage orientationImage(image.Shape().width, image.Shape().height, 1);

    // computeHarrisValues() computes the harris score at each pixel position, storing the
    // result in in harrisImage. 
    // You'll need to implement this function.
    computeHarrisValues(grayImage, harrisImage, orientationImage);

    // Threshold the harris image and compute local maxima.  You'll need to implement this function.
    computeLocalMaxima(harrisImage,harrisMaxImage);

    CByteImage tmp(harrisImage.Shape());
    convertToByteImage(harrisImage, tmp);
    WriteFile(tmp, "harris.tga");
    // WriteFile(harrisMaxImage, "harrisMax.tga");

    // Loop through feature points in harrisMaxImage and fill in information needed for 
    // descriptor computation for each point above a threshold. You need to fill in id, type, 
    // x, y, and angle.
    int id = 0;
    for (int y=0; y < harrisMaxImage.Shape().height; y++) {
        for (int x=0; x < harrisMaxImage.Shape().width; x++) {

            if (harrisMaxImage.Pixel(x, y, 0) == 0)
                continue;

            Feature f;

            //TODO: Fill in feature with location and orientation data here
            f.type = 2; // harris feature type = 2
            f.id = id; // id
            f.x = x;
            f.y = y;
            f.angleRadians = orientationImage.Pixel(x, y, 0); // TODO: need to figure this out

            features.push_back(f);
            id++;
        }
    }
}



//TO DO---------------------------------------------------------------------
// Loop through the image to compute the harris corner values as described in class
// srcImage:  grayscale of original image
// harrisImage:  populate the harris values per pixel in this image
void computeHarrisValues(CFloatImage &srcImage, CFloatImage &harrisImage, CFloatImage &orientationImage)
{
    printf("entering compute harris values\n");
    // std::cout << "srcImage nbands : " << srcImage.Shape().nBands << std::endl;
    int w = srcImage.Shape().width;
    int h = srcImage.Shape().height;

    // TODO: You may need to compute a few filtered images to start with
    CFloatImage blurImage(w, h, 1);
    CFloatImage xSobel(w, h, 1);
    CFloatImage ySobel(w, h, 1);
    // CFloatImage xxSobel(w, h, 1); // a
    // CFloatImage xySobel(w, h, 1); // b
    // CFloatImage yySobel(w, h, 1); // c

    int blurWindowSize = 5;
    CFloatImage blurKernel(blurWindowSize, blurWindowSize, 1);
    for(int row = 0; row < blurWindowSize; row++){
        for(int col = 0; col < blurWindowSize; col++){
            blurKernel.Pixel(row, col, 0) = gaussian5x5[row * blurWindowSize + col];
        }
    }
    blurKernel.origin[0] = blurWindowSize/2;
    blurKernel.origin[1] = blurWindowSize/2;

    // Blur the srcImage
    Convolve(srcImage, blurImage, blurKernel);

    // Calculate Gradient Images
    Convolve(srcImage, xSobel, ConvolveKernel_SobelX);
    Convolve(srcImage, ySobel, ConvolveKernel_SobelY);
    // Convolve(xSobel, xxSobel, ConvolveKernel_SobelX); // xxSobel = a
    // Convolve(xSobel, xySobel, ConvolveKernel_SobelY); // xySobel = b
    // Convolve(ySobel, yySobel, ConvolveKernel_SobelY); // yySobel = c

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            // TODO:  Compute the harris score for 'srcImage' at this pixel and store in 'harrisImage'.  See the project
            //   page for pointers on how to do this.  You should also store an orientation for each pixel in 
            //   'orientationImage'

            // 1. Calculate Harris Matrix for 5x5 window (a, b, c) 
            double a = 0.0;
            double b = 0.0;
            double c = 0.0;
            int idx = 0;
            for(int row = -2; row <= 2; row++){ // loop through 5x5 window
                for(int col = -2; col <= 2; col++){
                    double weight = gaussian5x5[idx /*((2+row) * 5) + (2+col)*/]; // circular gaussian weight
                    a += weight * (imagePixel(x+col, y+row, xSobel, 0)* imagePixel(x+col, y+row, xSobel, 0)); // deriv in x direction squared
                    b += weight * (imagePixel(x+col, y+row, xSobel, 0)* imagePixel(x+col, y+row, ySobel, 0)); // deriv in x and y direction
                    c += weight * (imagePixel(x+col, y+row, ySobel, 0)* imagePixel(x+col, y+row, ySobel, 0)); // deriv in y direction squared
                    idx++;
                }
            }

            // 2. Calculate "Corner Strength"
            double cornerScore = std::numeric_limits<double>::max();
            // double cornerScore = 0;
            if( (a + c) != 0){
                cornerScore = ((a * c) - (b * b)) / (a + c);
            }

            // 3. Assign "Corner Strength"
            harrisImage.Pixel(x,y,0) = cornerScore;

            // 4. Compute and Assign "Canonical Orientation" in radians (tan.inverse(dy/dx))
            double res = atan2(ySobel.Pixel(x,y,0), xSobel.Pixel(x,y,0));
            orientationImage.Pixel(x,y,0) = res;
        }
    }
    printf("exiting compute harris values\n");
}



//TO DO---------------------------------------------------------------------
//Loop through the image to compute the harris corner values as described in class
// srcImage:  image with Harris values
// destImage: Assign 1 to local maximum in 3x3 window, 0 otherwise
void computeLocalMaxima(CFloatImage &srcImage,CByteImage &destImage)
{
    printf("entering compute local maxima\n");
    int w = srcImage.Shape().width;
    int h = srcImage.Shape().height;
    std::cout << "width : " << w << ", height : " << h << std::endl;

    for(int y = 0 ; y < h ; y++){
        for(int x = 0 ; x < w ; x++){
            destImage.Pixel(x,y,0) = 0;
            double score = srcImage.Pixel(x,y,0);
            bool isMax = false;
            if(score > 0.011){ // is this above threshold?
                isMax = true;
            }

            for(int yy = -2 ; yy <= 2 ; yy++){
                for(int xx = -2 ; xx <= 2 ; xx++){
                    double localScore = imagePixel(x + xx, y + yy, srcImage, 0);
                    if( localScore > score){
                        isMax = false;
                    }
                }
            }

            if(isMax){
                destImage.Pixel(x,y,0) = 1;
            }
        }
    }
    printf("exiting compute local maxima\n");
}

// TODO: Implement parts of this function
// Compute Simple descriptors.
void ComputeSimpleDescriptors(CFloatImage &image, FeatureSet &features)
{
    std::cout << "Entering compute simple descriptors" << std::endl;
    //Create grayscale image used for Harris detection
    CFloatImage grayImage=ConvertToGray(image);

    const int windowSize = 5;
    CFloatImage destImage(windowSize, windowSize, 1);

    for (vector<Feature>::iterator i = features.begin(); i != features.end(); i++) {
        Feature &f = *i;

        int x = f.x;
        int y = f.y;

        f.data.resize(5 * 5);


        int range = 5 / 2;
        int index = 0;
        for (int row = -range; row <= range; row++ ){
            for (int col = -range; col <= range; col++){
                f.data[index] = imagePixel(x+col, y+row, grayImage, 0);
                index++;
            }
        }
    }
    std::cout << "Exiting compute simple descriptors" << std::endl;
}

// TODO: Implement parts of this function
// Compute MOPs descriptors.
void ComputeMOPSDescriptors(CFloatImage &image, FeatureSet &features)
{
    std::cout << "Entering mops" << std::endl;
    // This image represents the window around the feature you need to compute to store as the feature descriptor
    const int windowSize = 8;
    CFloatImage destImage(windowSize, windowSize, 1);

    // Gray image
    CFloatImage grayImage = ConvertToGray(image);

    // Blur Image
    CFloatImage blurImage(image.Shape().width, image.Shape().height, 1);

    // 40x40 Window
    CFloatImage bigWindow(40, 40, 1);

    // Blur 5x5 Gaussian kernel
    int blurWindowSize = 5;
    CFloatImage blurKernel(blurWindowSize, blurWindowSize, 1);
    for(int i = 0; i < blurWindowSize; i++){
        for(int j = 0; j < blurWindowSize; j++){
            blurKernel.Pixel(j,i,0) = gaussian5x5[blurWindowSize * i + j];
        }
    }
    blurKernel.origin[0] = blurWindowSize/2; // x offset to the center of the kernel
    blurKernel.origin[1] = blurWindowSize/2; // y offset to the center of the kernel
    Convolve(grayImage, blurImage, blurKernel);

    for (vector<Feature>::iterator i = features.begin(); i != features.end(); i++) {
        Feature &f = *i;

        //TODO: Compute the inverse transform as described by the feature location/orientation.
        //You'll need to compute the transform from each pixel in the 8x8 image 
        //to sample from the appropriate pixels in the 40x40 rotated window surrounding the feature

        // 1. Scale image by 1/5 (prefiltering)
        // 2. Rotate
        // 3. Sample 8x8 (already taken care of?)
        // 4. Intensity normalize the window by subtracting the mean, dividing by 
        //    the standard deviation in the window
        CTransform3x3 xform;

        CTransform3x3 origin;
        origin = origin.Translation( float(-windowSize/2), float(-windowSize/2));

        CTransform3x3 rotate;
        rotate = rotate.Rotation(f.angleRadians * 180.0 / PI);

        CTransform3x3 trans;
        trans = trans.Translation( f.x, f.y );

        CTransform3x3 scale;
        scale[0][0] = 40/windowSize;
        scale[1][1] = 40/windowSize;

        xform = trans * rotate * scale * origin; //origin * rotate * trans;

        WarpGlobal(blurImage, destImage, xform, eWarpInterpLinear);

        f.data.resize(windowSize * windowSize);

        // Normalize the patch
        double mean = 0.0;
        for (int row = 0; row < windowSize; row++ ){
            for (int col = 0; col < windowSize; col++){
                mean += destImage.Pixel(col, row, 0);
            }
        }
        mean = mean / (windowSize * windowSize);


        double variance = 0.0;
        for (int row = 0; row < windowSize; row++ ){
            for (int col = 0; col < windowSize; col++){
                variance += pow((destImage.Pixel(col, row, 0) - mean), 2);
            }
        }
        double stndiv = sqrt( variance / ( (windowSize * windowSize) - 1));

        // Fill in the feature descriptor data for a MOPS descriptor
        for (int row = 0; row < windowSize; row++ ){
            for (int col = 0; col < windowSize; col++){
                f.data[row * windowSize + col] = (destImage.Pixel(col, row, 0) - mean)/stndiv;
            }
        }
    }
    std::cout << "Exiting mops" << std::endl;
}

// Compute Custom descriptors (extra credit)
void ComputeCustomDescriptors(CFloatImage &image, FeatureSet &features)
{

}

// Perform simple feature matching.  This just uses the SSD
// distance between two feature vectors, and matches a feature in the
// first image with the closest feature in the second image.  It can
// match multiple features in the first image to the same feature in
// the second image.
void ssdMatchFeatures(const FeatureSet &f1, const FeatureSet &f2, vector<FeatureMatch> &matches) {
    int m = f1.size();
    int n = f2.size();

    matches.resize(m);

    double d;
    double dBest;
    int idBest;

    for (int i=0; i<m; i++) {
        dBest = 1e100;
        idBest = 0;

        for (int j=0; j<n; j++) {
            d = distanceSSD(f1[i].data, f2[j].data);

            if (d < dBest) {
                dBest = d;
                idBest = f2[j].id;
            }
        }

        matches[i].id1 = f1[i].id;
        matches[i].id2 = idBest;
        matches[i].distance = dBest;
    }
}

//TODO: Write this function to perform ratio feature matching.  
// This just uses the ratio of the SSD distance of the two best matches
// and matches a feature in the first image with the closest feature in the second image.
// It can match multiple features in the first image to the same feature in
// the second image.  (See class notes for more information)
// You don't need to threshold matches in this function -- just specify the match distance
// in each FeatureMatch object, as well as the ids of the two matched features (see
// ssdMatchFeatures for reference).
void ratioMatchFeatures(const FeatureSet &f1, const FeatureSet &f2, vector<FeatureMatch> &matches) 
{
    int f1Size = f1.size();
    int f2Size = f2.size();

    // Match indices
    int bestMatchIndex = 0;
    int nextBestMatchIndex = 0;

    double bestDistance = -1;
    double nextBestDistance = -1;

    // Distance
    double distance;

    // Loop over all features in FeatureSet 1
    for(int i = 0; i < f1Size; i++)
    {
        // Create new FeatureMatch object
        FeatureMatch newMatch;

        // Reset best match index and distance
        bestMatchIndex = 0;
        nextBestMatchIndex = 0;
        bestDistance = -1;
        nextBestDistance = -1;

        // Loop over all features in FeatureSet 2
        for(int j = 0; j < f2Size; j++)
        {
            // Calculate distance
            distance = distanceSSD(f1[i].data, f2[j].data);

            // Check if this is the smallest distance or first distance
            if(distance < bestDistance || bestDistance < 0)
            {

                nextBestMatchIndex = bestMatchIndex;
                nextBestDistance = bestDistance;

                bestMatchIndex = f2[j].id;
                bestDistance = distance;
            }
            // Else check if this is the second best distance
            else if(distance < nextBestDistance || nextBestDistance < 0)
            {
                nextBestMatchIndex = f2[j].id;
                nextBestDistance = distance;
            }
        }

        // Now we have the best and second best match
        // Check if these match distances pass the ratio test

        newMatch.id1 = f1[i].id;
        newMatch.id2 = bestMatchIndex;
        newMatch.distance = (bestDistance / nextBestDistance);

        matches.push_back(newMatch);
    }
}


// Convert Fl_Image to CFloatImage.
bool convertImage(const Fl_Image *image, CFloatImage &convertedImage) {
    if (image == NULL) {
        return false;
    }

    // Let's not handle indexed color images.
    if (image->count() != 1) {
        return false;
    }

    int w = image->w();
    int h = image->h();
    int d = image->d();

    // Get the image data.
    const char *const *data = image->data();

    int index = 0;

    for (int y=0; y<h; y++) {
        for (int x=0; x<w; x++) {
            if (d < 3) {
                // If there are fewer than 3 channels, just use the
                // first one for all colors.
                convertedImage.Pixel(x,y,0) = ((uchar) data[0][index]) / 255.0f;
                convertedImage.Pixel(x,y,1) = ((uchar) data[0][index]) / 255.0f;
                convertedImage.Pixel(x,y,2) = ((uchar) data[0][index]) / 255.0f;
            }
            else {
                // Otherwise, use the first 3.
                convertedImage.Pixel(x,y,0) = ((uchar) data[0][index]) / 255.0f;
                convertedImage.Pixel(x,y,1) = ((uchar) data[0][index+1]) / 255.0f;
                convertedImage.Pixel(x,y,2) = ((uchar) data[0][index+2]) / 255.0f;
            }

            index += d;
        }
    }

    return true;
}

// Convert CFloatImage to CByteImage.
void convertToByteImage(CFloatImage &floatImage, CByteImage &byteImage) {
    CShape sh = floatImage.Shape();

    assert(floatImage.Shape().nBands == byteImage.Shape().nBands);
    for (int y=0; y<sh.height; y++) {
        for (int x=0; x<sh.width; x++) {
            for (int c=0; c<sh.nBands; c++) {
                float value = floor(255*floatImage.Pixel(x,y,c) + 0.5f);

                if (value < byteImage.MinVal()) {
                    value = byteImage.MinVal();
                }
                else if (value > byteImage.MaxVal()) {
                    value = byteImage.MaxVal();
                }

                // We have to flip the image and reverse the color
                // channels to get it to come out right.  How silly!
                byteImage.Pixel(x,sh.height-y-1,sh.nBands-c-1) = (uchar) value;
            }
        }
    }
}

// Compute SSD distance between two vectors.
double distanceSSD(const vector<double> &v1, const vector<double> &v2) {
    int m = v1.size();
    int n = v2.size();

    if (m != n) {
        // Here's a big number.
        return 1e100;
    }

    double dist = 0;

    for (int i=0; i<m; i++) {
        dist += pow(v1[i]-v2[i], 2);
    }


    return sqrt(dist);
}

// Transform point by homography.
void applyHomography(double x, double y, double &xNew, double &yNew, double h[9]) {
    double d = h[6]*x + h[7]*y + h[8];

    xNew = (h[0]*x + h[1]*y + h[2]) / d;
    yNew = (h[3]*x + h[4]*y + h[5]) / d;
}

// Evaluate a match using a ground truth homography.  This computes the
// average SSD distance between the matched feature points and
// the actual transformed positions.
double evaluateMatch(const FeatureSet &f1, const FeatureSet &f2, const vector<FeatureMatch> &matches, double h[9]) {
    double d = 0;
    int n = 0;

    double xNew;
    double yNew;

    unsigned int num_matches = matches.size();
    for (unsigned int i=0; i<num_matches; i++) {
        int id1 = matches[i].id1;
        int id2 = matches[i].id2;
        applyHomography(f1[id1].x, f1[id1].y, xNew, yNew, h);
        d += sqrt(pow(xNew-f2[id2].x,2)+pow(yNew-f2[id2].y,2));
        n++;
    }	

    return d / n;
}

void addRocData(const FeatureSet &f1, const FeatureSet &f2, const vector<FeatureMatch> &matches, double h[9],
                vector<bool> &isMatch, double threshold, double &maxD) 
{
    double d = 0;

    double xNew;
    double yNew;

    unsigned int num_matches = matches.size();
    for (unsigned int i=0; i<num_matches; i++) {
        int id1 = matches[i].id1;
        int id2 = matches[i].id2;
        applyHomography(f1[id1].x, f1[id1].y, xNew, yNew, h);

        // Ignore unmatched points.  There might be a better way to
        // handle this.
        d = sqrt(pow(xNew-f2[id2].x,2)+pow(yNew-f2[id2].y,2));
        if (d<=threshold) {
            isMatch.push_back(1);
        } else {
            isMatch.push_back(0);
        }

        if (matches[i].distance > maxD)
            maxD = matches[i].distance;
    }	
}

vector<ROCPoint> computeRocCurve(vector<FeatureMatch> &matches,vector<bool> &isMatch,vector<double> &thresholds)
{
    vector<ROCPoint> dataPoints;

    for (int i=0; i < (int)thresholds.size();i++)
    {
        //printf("Checking threshold: %lf.\r\n",thresholds[i]);
        int tp=0;
        int actualCorrect=0;
        int fp=0;
        int actualError=0;
        int total=0;

        int num_matches = (int) matches.size();
        for (int j=0;j < num_matches;j++) {
            if (isMatch[j]) {
                actualCorrect++;
                if (matches[j].distance < thresholds[i]) {
                    tp++;
                }
            } else {
                actualError++;
                if (matches[j].distance < thresholds[i]) {
                    fp++;
                }
            }

            total++;
        }

        ROCPoint newPoint;
        //printf("newPoints: %lf,%lf",newPoint.trueRate,newPoint.falseRate);
        newPoint.trueRate=(double(tp)/actualCorrect);
        newPoint.falseRate=(double(fp)/actualError);
        //printf("newPoints: %lf,%lf",newPoint.trueRate,newPoint.falseRate);

        dataPoints.push_back(newPoint);
    }

    return dataPoints;
}

// Compute AUC given a ROC curve
double computeAUC(vector<ROCPoint> &results)
{
    double auc=0;
    double xdiff,ydiff;
    for (int i = 1; i < (int) results.size(); i++)
    {
        //fprintf(stream,"%lf\t%lf\t%lf\n",thresholdList[i],results[i].falseRate,results[i].trueRate);
        xdiff=(results[i].falseRate-results[i-1].falseRate);
        ydiff=(results[i].trueRate-results[i-1].trueRate);
        auc=auc+xdiff*results[i-1].trueRate+xdiff*ydiff/2;

    }
    return auc;
}

// Compute the distance between two features
double distanceAbs(const vector<double> &v1, const vector<double> &v2)
{
    int m = v1.size();
    int n = v2.size();

    if (m != n) {
        // Here's a big number.
        return 1e100;
    }

    double dist = 0;

    for (int i=0; i<m; i++) {
        //dist += abs(v1[i]-v2[i]);
    }


    return sqrt(dist);
}

double imagePixel(int x, int y, CFloatImage img, int band)
{
    int height = img.Shape().height;
    int width = img.Shape().width;
    int nBands = img.Shape().nBands;

    if(x < 0 || y < 0 || x >= width || y >= height || band >= nBands)
        return 0;

    else
        return img.Pixel(x, y, band);
}