package eus.ehu.tss;


import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.measure.ResultsTable;
import ij.process.ImageProcessor;
import inra.ijpb.label.LabelImages;
import inra.ijpb.measure.IntensityMeasures;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.concurrent.atomic.AtomicInteger;


/**
 * This class will hold the features that each superpixel region has.
 * This features will be the intensity measures from the MorphoLibJ library.
 */
public class RegionFeatures {

    public static int totalFeatures() {
        return Feature.numFeatures();
    }

    /**
     * enum that lists the Features that can be obtained from the MorphoLibJ Intensity Measures
     */
    public enum Feature{
        Mean("Mean"),
        Median("Median"),
        Mode("Mode"),
        Skewness("Skewness"),
        Kurtosis("Kurtosis"),
        StdDev("StdDev"),
        Max("Max"),
        Min("Min"),
        NeighborsMean("NeighborsMean"),
        NeighborsMedian("NeighborsMedian"),
        NeighborsMode("NeighborsMode"),
        NeighborsSkewness("NeighborsSkewness"),
        NeighborsKurtosis("NeighborsKurtosis"),
        NeighborsStdDev("NeighborsStdDev"),
        NeighborsMax("NeighborsMax"),
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
        int progress = 0;
        long startTime = System.currentTimeMillis();
        IntensityMeasures calculator = new IntensityMeasures(inputImage,labelImage);
        ArrayList<ResultsTable> results = new ArrayList<ResultsTable>();
        IJ.showProgress(progress,selectedFeatures.size());
        for (Feature selectedFeature : selectedFeatures) {
            IJ.showStatus("Calculating "+selectedFeature.label);
            switch (selectedFeature) {
                case Max:
                    results.add( calculator.getMax() );
                    break;
                case Min:
                    results.add( calculator.getMin() );
                    break;
                case Mean:
                    results.add( calculator.getMean() );
                    break;
                case Mode:
                    results.add( calculator.getMode() );
                    break;
                case Median:
                    results.add( calculator.getMedian() );
                    break;
                case StdDev:
                    results.add( calculator.getStdDev() );
                    break;
                case Kurtosis:
                    results.add( calculator.getKurtosis() );
                    break;
                case Skewness:
                    results.add( calculator.getSkewness() );
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
            IJ.showProgress(progress,selectedFeatures.size());
        }
        long elapsedTime = System.currentTimeMillis();
        long estimatedTime = System.currentTimeMillis() - startTime;
        IJ.log( "        Calculating features took " + estimatedTime + " ms");
        startTime = System.currentTimeMillis();
        ResultsTable mergedTable = new ResultsTable();
        final int numLabels = results.get( 0 ).getCounter();
        for(int i=0; i < numLabels; ++i)
        {
            mergedTable.incrementCounter();
            String label = results.get( 0 ).getLabel( i );
            mergedTable.addLabel(label);

            for (ResultsTable result : results) {
                String measure = result.getColumnHeading(0);
                double value = result.getValue(measure, i);
                if(!Double.isFinite(value)&&measure.equals("Skewness")){
                    value=0;
                }else if(!Double.isFinite(value)&&measure.equals("Kurtosis")){
                    value=-1.2;
                }
                mergedTable.addValue(measure, value);
            }
        }
        elapsedTime = System.currentTimeMillis();
        estimatedTime = System.currentTimeMillis() - startTime;
        IJ.log( "        Merging features took " + estimatedTime + " ms");
        startTime = System.currentTimeMillis();
        //mergedTable.show( inputImage.getShortTitle() + "-intensity-measurements" );
        ArrayList<Attribute> attributes = new ArrayList<Attribute>();
        int numFeatures = mergedTable.getLastColumn()+1; //Take into account it starts in index 0
        for(int i=0;i<numFeatures;++i){
            attributes.add(new Attribute(mergedTable.getColumnHeading(i),i));
        }
        attributes.add(new Attribute("Class", classes));
        Instances unlabeled = new Instances("training data",attributes,0);
        for(int i=0;i<mergedTable.size();++i){
            //numFeatures is the index, add 1 to get number of attributes needed plus class
            Instance inst = new DenseInstance(numFeatures+1);
            for(int j=0;j<numFeatures;++j){
                inst.setValue(j,mergedTable.getValueAsDouble(j,i));
            }
            inst.setValue( numFeatures, 0 );//set class as 0
            unlabeled.add(inst);
        }
        unlabeled.setClassIndex( numFeatures );
        elapsedTime = System.currentTimeMillis();
        estimatedTime = System.currentTimeMillis() - startTime;
        IJ.log( "        Setting class label as 0 took " + estimatedTime + " ms");
        //The number or instances should be the same as the size of the table
        if(unlabeled.numInstances()!=(mergedTable.size())){
            return null;
        }else{
            return unlabeled;
        }
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
        int progress = 0;
        long startTime = System.currentTimeMillis();
        HashMap<Integer, int[]> labelCoord = calculateLabelCoordinates(labelImage);
        ImageStack gtStack = gtImage.getImageStack();
        IntensityMeasures calculator = new IntensityMeasures(inputImage,labelImage);
        ArrayList<ResultsTable> results = new ArrayList<ResultsTable>();
        IJ.showProgress(progress,selectedFeatures.size());
        for (Feature selectedFeature : selectedFeatures) {
            IJ.showStatus("Calculating "+selectedFeature.label);
            switch (selectedFeature) {
                case Max:
                    results.add( calculator.getMax() );
                    break;
                case Min:
                    results.add( calculator.getMin() );
                    break;
                case Mean:
                    results.add( calculator.getMean() );
                    break;
                case Mode:
                    results.add( calculator.getMode() );
                    break;
                case Median:
                    results.add( calculator.getMedian() );
                    break;
                case StdDev:
                    results.add( calculator.getStdDev() );
                    break;
                case Kurtosis:
                    results.add( calculator.getKurtosis() );
                    break;
                case Skewness:
                    results.add( calculator.getSkewness() );
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
            IJ.showProgress(progress,selectedFeatures.size());
        }
        long elapsedTime = System.currentTimeMillis();
        long estimatedTime = System.currentTimeMillis() - startTime;
        IJ.log( "        Calculating features took " + estimatedTime + " ms");
        startTime = System.currentTimeMillis();
        ResultsTable mergedTable = new ResultsTable();
        final int numLabels = results.get( 0 ).getCounter();
        for(int i=0; i < numLabels; ++i)
        {
            mergedTable.incrementCounter();
            String label = results.get( 0 ).getLabel( i );
            mergedTable.addLabel(label);

            for (ResultsTable result : results) {
                String measure = result.getColumnHeading(0);
                double value = result.getValue(measure, i);
                if(!Double.isFinite(value)&&measure.equals("Skewness")){
                    value=0;
                }else if(!Double.isFinite(value)&&measure.equals("Kurtosis")){
                    value=-1.2;
                }
                mergedTable.addValue(measure, value);
            }
        }
        elapsedTime = System.currentTimeMillis();
        estimatedTime = System.currentTimeMillis() - startTime;
        IJ.log( "        Merging features took " + estimatedTime + " ms");
        startTime = System.currentTimeMillis();
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
        elapsedTime = System.currentTimeMillis();
        estimatedTime = System.currentTimeMillis() - startTime;
        IJ.log( "        Setting class label as 0 took " + estimatedTime + " ms");
        //The number or instances should be the same as the size of the table
        if(labeled.numInstances()!=(mergedTable.size())){
            return null;
        }else{
            return labeled;
        }
    }

    /**
     * Calculates coordinates corresponding to labels in label image
     * @param labelImage input image with labels
     * @return a HashMap where the key is the label and the values are the coordinates of the label
     */
    private static HashMap<Integer,int[]> calculateLabelCoordinates(ImagePlus labelImage){
        HashMap<Integer, Integer> labelIndices = null;
        HashMap<Integer, int[]> result = new HashMap<>();
        final int width = labelImage.getWidth();
        final int height = labelImage.getHeight();

        int[] labels = LabelImages.findAllLabels(labelImage.getImageStack());
        int numLabels = labels.length;
        labelIndices = LabelImages.mapLabelIndices(labels);
        final int numSlices = labelImage.getImageStackSize();
        for( int z=1; z <= numSlices; z++ )
        {
            final ImageProcessor labelsIP = labelImage.getImageStack().getProcessor( z );

            for( int x = 0; x<width; x++ )
                for( int y = 0; y<height; y++ )
                {
                    int labelValue = (int) labelsIP.getf( x, y );
                    int[] coord = new int[3];
                    coord[0] = x; coord[1] = y; coord[2] = z;
                    result.putIfAbsent(labelValue,coord);
                }
        }
        return result;
    }


}
