package eus.ehu.superpixel.learning.supervised;


import ij.ImagePlus;
import ij.measure.ResultsTable;
import inra.ijpb.measure.IntensityMeasures;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.awt.*;
import java.util.ArrayList;


/**
 * This class will hold the features that each superpixel region has.
 * This features will be the intensity measures from the MorphoLibJ library.
 */
public class RegionFeatures {

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
        Min("Min");

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
            int n=0;
            for(String item : getAllLabels()){
                n++;
            }
            return n;
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
     * @return ResultsTable with the features of each region from the labelImage
     */
    public static Instances calculateRegionFeatures(ImagePlus inputImage, ImagePlus labelImage, ArrayList<Feature> selectedFeatures){
        IntensityMeasures calculator = new IntensityMeasures(inputImage,labelImage);
        ArrayList<ResultsTable> results = new ArrayList<ResultsTable>();
        for (Feature selectedFeature : selectedFeatures) {
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
            }
        }

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
                mergedTable.addValue(measure, value);
            }
        }
        mergedTable.show( inputImage.getShortTitle() + "-intensity-measurements" );
        ArrayList<Attribute> attributes = new ArrayList<Attribute>();
        int numFeatures = mergedTable.getLastColumn(); //Take into account it starts in index 0
        for(int i=0;i<numFeatures+1;++i){
            attributes.add(new Attribute(mergedTable.getColumnHeading(i),i));
        }
        attributes.add(new Attribute("Class"));
        Instances unlabeled = new Instances("dataset",attributes,0);
        for(int i=0;i<mergedTable.size();++i){
            Instance inst = new DenseInstance(numFeatures+2);//numFeatures is the index, add 1 to get number of attributes needed plus class
            for(int j=0;j<(numFeatures+1);++j){
                inst.setValue(j,mergedTable.getValue(j,i));
            }
            inst.setValue(numFeatures+1,0);//set class as 0
            unlabeled.add(inst);
        }
        unlabeled.setClassIndex(numFeatures+1);
        return unlabeled;
    }


}
