package eus.ehu.tss;


import ij.IJ;
import ij.ImagePlus;
import ij.measure.ResultsTable;
import ij.process.ImageProcessor;
import inra.ijpb.measure.IntensityMeasures;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.HashMap;


/**
 * This class will hold the features that each superpixel region has.
 * This features will be the intensity measures from the MorphoLibJ library.
 */
public class RegionFeatures {

    /**
     * Get total number of features.
     * @return total number of features.
     */
    public static int totalFeatures() {
        return Feature.numFeatures();
    }

    /**
     * enum that lists the Features that can be obtained from the MorphoLibJ Intensity Measures
     */
    public enum Feature{
        /** Mean intensity value in the region */
        Mean("Mean"),
        /** Median intensity value in the region */
        Median("Median"),
        /** Mode of the intensity in the region */
        Mode("Mode"),
        /** Skewness of the intensity in the region */
        Skewness("Skewness"),
        /** Kurtosis of the intensity in the region */
        Kurtosis("Kurtosis"),
        /** Standard deviation of the intensity in the region */
        StdDev("StdDev"),
        /** Maximum value of intensity in the region */
        Max("Max"),
        /** Minimum value of intensity in the region */
        Min("Min"),
        /** Mean intensity value in the neighbor regions */
        NeighborsMean("NeighborsMean"),
        /** Median intensity value in the neighbor regions */
        NeighborsMedian("NeighborsMedian"),
        /** Mode of the intensity value in the neighbor regions */
        NeighborsMode("NeighborsMode"),
        /** Skewness of the intensity in the neighbor regions */
        NeighborsSkewness("NeighborsSkewness"),
        /** Kurtosis of the intensity in the neighbor regions */
        NeighborsKurtosis("NeighborsKurtosis"),
        /** Standard deviation of the intensity in the neighbor regions */
        NeighborsStdDev("NeighborsStdDev"),
        /** Maximum value of the intensity in the neighbor regions */
        NeighborsMax("NeighborsMax"),
        /** Minimum value of the intensity in the neighbor regions */
        NeighborsMin("NeighborsMin");


        private final String label;

        /**
         * Create feature with a label
         * @param label String with the label of the feature to be created
         */
        private Feature(String label) {this.label = label;}

        /**
         * Returns string with the value of the label
         * @return String with the value of the label
         */
        public String toString(){ return this.label;}

        /**
         * Gets a String array with all labels that are listed in this enum
         * @return String array with values of all labels
         */
        public static String[] getAllLabels(){
            int n = Feature.values().length;
            String[] result = new String[n];
            int i=0;
            for(Feature feature : Feature.values()){
                result[i++] = feature.label;
            }
            return result;
        }

        /**
         * Gets the number of features that are in this enum
         * @return int with number of features that are in this enum
         */
        public static int numFeatures(){
            return getAllLabels().length;
        }

        /**
         * Based on provided label returns Feature with that labelx
         * @param fLabel String with the name of the feature
         * @return Feature which matches the provided String
         */
        public static Feature fromLabel(String fLabel){
            if(fLabel != null){
                fLabel = fLabel.toLowerCase();
                for(Feature feature : Feature.values()){
                    String cmp = feature.label.toLowerCase();
                    if(cmp.equals(fLabel)){
                        return feature;
                    }
                }
                throw new IllegalArgumentException("Unable to parse Feature with label " + fLabel);
            }
            return null;
        }

    };

    /**
     * Calculate Features for each region of label image
     * @param inputImage input image
     * @param labelImage corresponding label image
     * @param selectedFeatures ArrayList of selected features
     * @return table with features per region
     */
    public static ResultsTable calculateFeaturesTable(
            ImagePlus inputImage,
            ImagePlus labelImage,
            ArrayList<Feature> selectedFeatures)
    {
        IntensityMeasures calculator = new IntensityMeasures(inputImage, labelImage);
        int progress = 0;
        ArrayList<ResultsTable> results = new ArrayList<ResultsTable>();
        for (Feature selectedFeature : selectedFeatures) {
            IJ.showStatus("Calculating " + selectedFeature.label);
            switch (selectedFeature) {
                case Max:
                    results.add(calculator.getMax());
                    break;
                case Min:
                    results.add(calculator.getMin());
                    break;
                case Mean:
                    results.add(calculator.getMean());
                    break;
                case Mode:
                    results.add(calculator.getMode());
                    break;
                case Median:
                    results.add(calculator.getMedian());
                    break;
                case StdDev:
                    results.add(calculator.getStdDev());
                    break;
                case Kurtosis:
                    results.add(calculator.getKurtosis());
                    break;
                case Skewness:
                    results.add(calculator.getSkewness());
                    break;
                case NeighborsMean:
                    results.add(calculator.getNeighborsMean());
                    break;
                case NeighborsMedian:
                    results.add(calculator.getNeighborsMedian());
                    break;
                case NeighborsMode:
                    results.add(calculator.getNeighborsMode());
                    break;
                case NeighborsSkewness:
                    results.add(calculator.getNeighborsSkewness());
                    break;
                case NeighborsKurtosis:
                    results.add(calculator.getNeighborsKurtosis());
                    break;
                case NeighborsStdDev:
                    results.add(calculator.getNeighborsStdDev());
                    break;
                case NeighborsMax:
                    results.add(calculator.getNeighborsMax());
                    break;
                case NeighborsMin:
                    results.add(calculator.getNeighborsMin());
                    break;
            }
            ++progress;
            IJ.showProgress(progress, selectedFeatures.size());
        }
        ResultsTable mergedTable = new ResultsTable();
        final int numLabels = results.get(0).getCounter();
        for (int i = 0; i < numLabels; ++i) {
            mergedTable.incrementCounter();
            String label = results.get(0).getLabel(i);
            mergedTable.addLabel(label);

            for (ResultsTable result : results) {
                String measure = result.getColumnHeading(0);
                double value = result.getValue(measure, i);
                if (!Double.isFinite(value) && measure.equals("Skewness")) {
                    value = 0;
                } else if (!Double.isFinite(value) && measure.equals("Kurtosis")) {
                    value = -1.2;
                }
                mergedTable.addValue(measure, value);
            }
        }
        return mergedTable;

    }

    /**
     * Calculate Instances based on provided table
     * @param resultsTable table with features
     * @param classes possible classes
     * @return resulting Instances
     */
    public static Instances calculateUnabeledInstances(ResultsTable resultsTable, ArrayList<String> classes){
        ArrayList<Attribute> attributes = new ArrayList<Attribute>();
        int numFeatures = resultsTable.getLastColumn() + 1; //Take into account it starts in index 0
        for (int i = 0; i < numFeatures; ++i) {
            attributes.add(new Attribute(resultsTable.getColumnHeading(i), i));
        }
        attributes.add(new Attribute("Class", classes));
        Instances unlabeled = new Instances("training data", attributes, 0);
        for (int i = 0; i < resultsTable.size(); ++i) {
            //numFeatures is the index, add 1 to get number of attributes needed plus class
            Instance inst = new DenseInstance(numFeatures + 1);
            for (int j = 0; j < numFeatures; ++j) {
                inst.setValue(j, resultsTable.getValueAsDouble(j, i));
            }
            inst.setValue(numFeatures, 0);//set class as 0
            unlabeled.add(inst);
        }
        unlabeled.setClassIndex(numFeatures);
        //The number or instances should be the same as the size of the table
        if (unlabeled.numInstances() != (resultsTable.size())) {
            return null;
        }
        return unlabeled;
    }


    /**
     * Calculates the selected features of each region based on an input image and a labeled image
     * @param inputImage ImagePlus input image from which the features will be calculated
     * @param labelImage ImagePlus where the labels are located
     * @param selectedFeatures ArrayList of Feature with the features that need to be calculated
     * @param classes list with the class names to use
     * @return dataset with the features of each region from the labelImage
     */
    public static Instances calculateUnlabeledRegionFeatures(
    		ImagePlus inputImage,
    		ImagePlus labelImage,
    		ArrayList<Feature> selectedFeatures,
    		ArrayList<String> classes)
    {
        ResultsTable resultsTable = calculateFeaturesTable(inputImage,labelImage,selectedFeatures);
        Instances unlabeled = calculateUnabeledInstances(resultsTable,classes);
        return unlabeled;
    }

    /**
     * Calculates the selected features of each region based on an input image, a labeled image and a ground truth image
     * @param inputImage ImagePlus input image from which the features will be calculated
     * @param labelImage ImagePlus where the labels are located
     * @param gtImage ImagePlus that provides Ground Truth
     * @param selectedFeatures ArrayList of Feature with the features that need to be calculated
     * @param classes list with the class names to use
     * @return dataset with the features of each region from the labelImage
     */
    public static Instances calculateLabeledRegionFeatures(
            ImagePlus inputImage,
            ImagePlus labelImage,
            ImagePlus gtImage,
            ArrayList<Feature> selectedFeatures,
            ArrayList<String> classes)
    {
        HashMap<Integer, int[]> labelCoord = Utils.calculateLabelCoordinates(labelImage);
        ResultsTable mergedTable = calculateFeaturesTable(inputImage,labelImage,selectedFeatures);
        //mergedTable.show( inputImage.getShortTitle() + "-intensity-measurements" );
        ArrayList<Attribute> attributes = new ArrayList<Attribute>();
        int numFeatures = mergedTable.getLastColumn()+1; //Take into account it starts in index 0
        for(int i=0;i<numFeatures;++i){
            attributes.add(new Attribute(mergedTable.getColumnHeading(i),i));
        }
        attributes.add(new Attribute("Class", classes));
        Instances labeled = new Instances("training data",attributes,0);
        for(int i=0;i<mergedTable.size();++i){
            //numFeatures is the index, add 1 to get number of attributes needed plus class
            Instance inst = new DenseInstance(numFeatures+1);
            for(int j=0;j<numFeatures;++j){
                inst.setValue(j,mergedTable.getValueAsDouble(j,i));
            }
            int[] coord = labelCoord.get(i+1);
            ImageProcessor gtProcessor = gtImage.getProcessor();
            float value = (float) gtProcessor.getf(coord[0],coord[1]);
            inst.setValue( numFeatures, (int) value );
            labeled.add(inst);
        }
        labeled.setClassIndex( numFeatures );
        //The number or instances should be the same as the size of the table
        if(labeled.numInstances()!=(mergedTable.size())){
            return null;
        }else{
            return labeled;
        }
    }



}
