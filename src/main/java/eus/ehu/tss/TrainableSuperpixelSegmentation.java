package eus.ehu.tss;

import ij.ImagePlus;
import ij.ImageStack;
import ij.measure.ResultsTable;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;
import inra.ijpb.label.LabelImages;
import inra.ijpb.measure.ResultsBuilder;
import weka.classifiers.AbstractClassifier;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;

import java.util.ArrayList;
import java.util.HashMap;

/**
 * Main class of the library that will conduct the classification of the images.
 */
public class TrainableSuperpixelSegmentation {

    private ArrayList<RegionFeatures.Feature> selectedFeatures = new ArrayList<RegionFeatures.Feature>();
    private ImagePlus inputImage;
    private ImagePlus labelImage;
    private ImagePlus resultImage;
    private Instances trainingData = null;
    private AbstractClassifier abstractClassifier;
    private ResultsTable unlabeledTable=null;
    private boolean classifierTrained = false;
    private boolean balanceClasses = true;
    ArrayList<String> classes = null;

    /**
     * Empty builder
     */
    public TrainableSuperpixelSegmentation(){

    }


    /**
     * Creates instance of TrainableSuperpixelSegmentation based on an image and it's corresponding label image, a list of selected features and an AbstractClassifier
     * @param originalImage ImagePlus image that will be analyzed
     * @param labels ImagePlus labeled image of the originalImage
     * @param features ArrayList of Features (from RegionFeatures.Feature) that represent the features that will be calculated
     * @param classifier AbstractClassifier that will be used to classify the images
     */
    public TrainableSuperpixelSegmentation(
    		ImagePlus originalImage,
    		ImagePlus labels,
    		ArrayList<RegionFeatures.Feature> features,
    		AbstractClassifier classifier,
    		ArrayList<String> classes )
    {
        selectedFeatures = features;
        inputImage = originalImage;
        labelImage = labels;
        abstractClassifier  = classifier;
        this.classes = classes;
    }

    /**
     * Calculates the selected features for each region and saves them on the private variable unlabeled
     * @return boolean that checks if the region features have been created
     */
    public boolean calculateRegionFeatures(){
        if(inputImage.getType()==ImagePlus.COLOR_RGB){
            unlabeledTable = RegionColorFeatures.calculateFeaturesTable(
                    inputImage,
                    labelImage,
                    selectedFeatures);
        }else {
            unlabeledTable = RegionFeatures.calculateFeaturesTable(
                    inputImage,
                    labelImage,
                    selectedFeatures);
        }
        return unlabeledTable != null;
    }


    /**
     * Returns a String with ARFF format of the features for each region
     * @return String in ARFF format
     */
    public String getFeaturesByRegion(){
        Instances unlabeled = RegionFeatures.calculateUnabeledInstances(unlabeledTable,classes);
        return unlabeled.toString();
    }

    /**
     * Calculates training data based on provided region list
     * @param classRegions ArrayList of int[] where each int[] represents the labels of superpixels that belong to the class indicated by their index in the ArrayList
     * @return boolean value false when training has had an error
     */
    public boolean calculateTrainingData(ArrayList<int[]> classRegions){
        // read attributes from unlabeled data
        ArrayList<Attribute> attributes = new ArrayList<Attribute>();
        if(unlabeledTable==null){
            calculateRegionFeatures();
        }
        int numFeatures = unlabeledTable.getLastColumn()+ 1;
        for(int i=0;i<numFeatures;++i){
            attributes.add(new Attribute(unlabeledTable.getColumnHeading(i),i));
        }
        attributes.add(new Attribute("Class",classes));
        Instances newTrainingData = new Instances("training data",attributes,0);
        int[] labels = LabelImages.findAllLabels(labelImage);
        HashMap<Integer,Integer> labelIndices = LabelImages.mapLabelIndices(labels);

        for(int i=0;i<classRegions.size();++i){ //For each class in classRegions
            for(int j=0;j<classRegions.get(i).length;++j){
                Instance inst = new DenseInstance(numFeatures+1);
                for(int k=0;k<numFeatures;++k){
                    int classvalue = classRegions.get(i)[j];
                    inst.setValue(k,unlabeledTable.getValueAsDouble(k,
                            labelIndices.get(
                                    classRegions.get(i)[j]
                            )));
                }
                inst.setValue(numFeatures,i); // set class value
                newTrainingData.add(inst);
            }
        }
        newTrainingData.setClassIndex(numFeatures); // set class index
        if(trainingData!=null){
            try {
                trainingData = Utils.merge(trainingData, newTrainingData);
            }catch (Exception e){
                e.printStackTrace();
                return false;
            }
        }else {
            trainingData = newTrainingData;
        }
        return true;
    }

    /**
     * Train classifier based on previously loaded training data
     * @return boolean value false when training has had an error
     */
    public boolean trainClassifier(){
        try {
            if(trainingData==null){
                System.out.println("Add training data for training");
                return false;
            }
            if(balanceClasses){
                try {
                    final Resample filter = new Resample();
                    filter.setBiasToUniformClass(1.0);
                    filter.setInputFormat(trainingData);
                    filter.setNoReplacement(false);
                    filter.setSampleSizePercent(100);
                    trainingData = Filter.useFilter(trainingData, filter);
                }catch (Exception e){
                    e.printStackTrace();
                }
            }
            abstractClassifier.buildClassifier(trainingData);
            classifierTrained = true;
            return true;
        } catch (Exception e) {
            System.out.println("Error when building classifier");
            e.printStackTrace();
            return false;
        }
    }

    /**
     * Applies classifier to unlabeled data and creates new Instances in labeled private variables
     * @return ImagePlus with classified image
     */
    public ImagePlus applyClassifier(){
        try {
            if(unlabeledTable==null){
                calculateRegionFeatures();
            }
            if(!classifierTrained){
                if(!trainClassifier()){
                    System.out.println("Error when training classifier");
                    return null;
                }
            }
            int numAttributes = unlabeledTable.getLastColumn();
            ResultsBuilder resultsBuilder = new ResultsBuilder((ResultsTable) unlabeledTable.clone());
            ResultsTable classesTable = new ResultsTable();
            ArrayList<Attribute> attributes = new ArrayList<Attribute>();
            for(int i=0;i<numAttributes;++i){
                attributes.add(new Attribute(unlabeledTable.getColumnHeading(i),i));
            }
            attributes.add(new Attribute("Class", classes));
            Instances labeled = new Instances("Labeled",attributes,0);
            labeled.setClassIndex(numAttributes);
            double[] values = new double[unlabeledTable.getCounter()];
            for(int i=0;i<unlabeledTable.getCounter();++i){
                Instance ins = new DenseInstance(numAttributes+1);//+1 for class attribute
                for(int j=0;j<numAttributes;++j){
                    ins.setValue(j,unlabeledTable.getValueAsDouble(j,i));
                }
                ins.setDataset(labeled);
                double classLabel = abstractClassifier.classifyInstance(ins);
                values[i]=classLabel;
                classesTable.incrementCounter();
                classesTable.addLabel(unlabeledTable.getLabel(i));
                classesTable.addValue("Class",classLabel);
            }
            resultsBuilder.addResult(classesTable);
            ImageStack res = LabelImages.applyLut(labelImage.getImageStack(),values);
            ImagePlus result = new ImagePlus(inputImage.getShortTitle()+"-classified",res);
            return result;
        } catch (Exception e) {
            System.out.println("Error when applying classifier");
            e.printStackTrace();
            return null;
        }
    }

    /**
     * Applies already trained classifier to input image and label image
     * @param inImage input image
     * @param lbImage superpixel segmentated label image
     * @return classified image
     */
    public ImagePlus applyClassifier(ImagePlus inImage, ImagePlus lbImage){
        if(!classifierTrained){
            if(!trainClassifier()){
                System.out.println("Error when training classifier");
                return null;
            }
        }
        inputImage = inImage;
        labelImage = lbImage;
        if(calculateRegionFeatures()){
            return applyClassifier();
        }else {
            System.out.println("Error when calculating region features");
            return inImage;
        }

    }

    /**
     * Returns a probability map image
     * @return
     */
    public ImagePlus getProbabilityMap(){
        if(unlabeledTable==null){
            calculateRegionFeatures();
        }
        if(!classifierTrained){
            if(!trainClassifier()){
                System.out.println("Error when training classifier");
                return null;
            }
        }
        Instances unlabeled = RegionFeatures.calculateUnabeledInstances(unlabeledTable,classes);
        final int numInstances = unlabeled.numInstances();
        final int numClasses = classes.size();
        final int width = labelImage.getWidth();
        final int height = labelImage.getHeight();
        final double[][] classificationResult = new double[numClasses][numInstances];
        ImageStack classificationResultImage = new ImageStack(width,height);
        for(int i=0;i<numInstances;++i){
            try{
                double[] prob = abstractClassifier.distributionForInstance(unlabeled.get(i));
                for(int k=0; k<numClasses;++k){
                    classificationResult[k][i] = prob[k];
                }
            }catch (Exception e){
                System.out.println("Could not apply Classifier!");
                e.printStackTrace();
                return null;
            }
        }
        ImageStack stackLabels = labelImage.getStack();
        double tags[] = new double[height*width];
        for(int k = 0;k<numClasses;++k) {
            for (int slice = 1; slice <= inputImage.getNSlices(); ++slice) {
                ImageProcessor ip = stackLabels.getProcessor(slice);
                for (int x = 0; x < width; ++x) {
                    for (int y = 0; y < height; ++y) {
                        int index = (int) ip.getPixelValue(x, y);
                        if (index == 0) { //edge pixel
                            tags[x + y * width] = index;
                        } else {
                            tags[x + y * width] = classificationResult[k][index-1];
                        }
                    }
                }
                FloatProcessor processor = new FloatProcessor(width, height,tags);
                classificationResultImage.addSlice(classes.get(k),processor.duplicate());
            }
        }
        ImagePlus result = new ImagePlus(inputImage.getShortTitle()+"-probMap",classificationResultImage);
        return result;

    }

    public ImagePlus getFeatureImage(ImagePlus labelImage, ResultsTable features){
        int columns = features.getLastColumn();
        ImageStack imageStack = new ImageStack(labelImage.getWidth(),labelImage.getHeight());
        for(int i=0;i<columns;++i){
            double[] values = new double[features.getCounter()];
            for(int j=1;j<features.getCounter();++j){
                values[j]=features.getValueAsDouble(i,j);
            }
            ImageStack stack = LabelImages.applyLut(labelImage.getImageStack(),values);
            ImageProcessor ip = stack.getProcessor(1);
            imageStack.addSlice(features.getColumnHeading(i),ip);
        }
        return new ImagePlus("Feature Image Stack",imageStack);
    }

    /**
     * Set a classifier
     * @param classifier input classifier
     * @return true on success and false on failure
     */
    public boolean setClassifier(AbstractClassifier classifier){
        if(classifier!=null){
            abstractClassifier = classifier;
            return true;
        }
        return false;
    }

    /**
     * Returns classifier
     * @return classifier
     */
    public AbstractClassifier getClassifier(){
        return abstractClassifier;
    }

    /**
     * Returns labeled instances
     * @return
     */
    public Instances getInstances() {
        return RegionFeatures.calculateUnabeledInstances(unlabeledTable,classes);
    }

    /**
     * Return result image
     * @return result image
     */
    public ImagePlus getResultImage() {
        return resultImage;
    }

    /**
     * Returns training data
     * @return training data
     */
    public Instances getTrainingData() {
        return trainingData;
    }


    /**
     * Returns list of classes
     * @return list of classes
     */
    public ArrayList<String> getClasses() {
        return classes;
    }

    /**
     * Sets list of classes
     * @param classes ArrayList of class names
     */
    public void setClasses(ArrayList<String> classes) {
        this.classes = classes;
    }

    /**
     * Adds features
     * @param features
     */
    public void addFeatures(String[] features){
        if(selectedFeatures==null){
            selectedFeatures = new ArrayList<RegionFeatures.Feature>();
            for(int i=0; i<features.length;++i){
                if(!selectedFeatures.contains(RegionFeatures.Feature.fromLabel(features[i]))){
                    selectedFeatures.add(RegionFeatures.Feature.fromLabel(features[i]));
                }
            }for(int i=0; i<features.length;++i){
                if(!selectedFeatures.contains(RegionFeatures.Feature.fromLabel(features[i]))){
                    selectedFeatures.add(RegionFeatures.Feature.fromLabel(features[i]));
                }
            }
        }else{
            for(int i=0; i<features.length;++i){
                if(!selectedFeatures.contains(RegionFeatures.Feature.fromLabel(features[i]))){
                    selectedFeatures.add(RegionFeatures.Feature.fromLabel(features[i]));
                }
            }
        }
    }


    /**
     * Sets selected features
     * @param selectedFeatures ArrayList of RegionFeatures.Feature elements
     */
    public void setSelectedFeatures(ArrayList<RegionFeatures.Feature> selectedFeatures) {
        this.selectedFeatures = selectedFeatures;
    }

    /**
     * Sets training data
     * @param trainingData Instances to be used in the training
     */
    public void setTrainingData(Instances trainingData) {
        this.trainingData = trainingData;
    }

    /**
     * Returns provided input image
     * @return ImagePlus with input image
     */
    public ImagePlus getInputImage() {
        return inputImage;
    }


    /**
     * Set input image
     * @param inputImage
     */
    public void setInputImage(ImagePlus inputImage) {
        this.inputImage = inputImage;
    }

    /**
     * Returns superpixel image with labels
     * @return
     */
    public ImagePlus getLabelImage() {
        return labelImage;
    }

    /**
     * Sets superpixel image
     * @param labelImage
     */
    public void setLabelImage(ImagePlus labelImage) {
        this.labelImage = labelImage;
    }

    /**
     * Returns wether a classifier has been trained
     * @return
     */
    public boolean isClassifierTrained() {
        return classifierTrained;
    }

    /**
     * Sets classifier as trained if true and as not trained if false
     * @param isClassifierTrained
     */
    public void setClassifierTrained(boolean isClassifierTrained){
        classifierTrained = isClassifierTrained;
    }

    /**
     * Returns unlabeled instances
     * @return
     */
    public Instances getUnlabeled() {
        return RegionFeatures.calculateUnabeledInstances(unlabeledTable,classes);
    }

    /**
     * Returns selected feature ArrayList
     * @return
     */
    public ArrayList<RegionFeatures.Feature> getSelectedFeatures() {
        return selectedFeatures;
    }

    /**
     * Set balancing of classes
     * @param balanceClasses
     */
    public void setBalanceClasses(boolean balanceClasses) {
        this.balanceClasses = balanceClasses;
    }


    public ResultsTable getUnlabeledTable() {
        return unlabeledTable;
    }

    public void setUnlabeledTable(ResultsTable unlabeledTable) {
        this.unlabeledTable = unlabeledTable;
    }




}
