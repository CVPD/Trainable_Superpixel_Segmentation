package eus.ehu.tss;

import ij.IJ;
import ij.ImagePlus;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.RandomForest;

import java.util.ArrayList;


public class TestColorSuperpixelSegmentation {
    public static void main(final String[] args){
        ImagePlus inputImage = IJ.openImage( TestSuperpixelSegmentation.class.getResource( "/TMA3d.tif" ).getFile() );
        inputImage.show();
        ImagePlus labelImage = IJ.openImage( TestSuperpixelSegmentation.class.getResource( "/TMA-Segmentation-16-bit3d.tif").getFile() );
        labelImage.show();

        // Use all features
        ArrayList<RegionFeatures.Feature> selectedFeatures = new ArrayList<>();
        selectedFeatures.add(RegionFeatures.Feature.fromLabel("Mean"));
        selectedFeatures.add(RegionFeatures.Feature.fromLabel("Median"));
        selectedFeatures.add(RegionFeatures.Feature.fromLabel("Mode"));
        selectedFeatures.add(RegionFeatures.Feature.fromLabel("Skewness"));
        selectedFeatures.add(RegionFeatures.Feature.fromLabel("Kurtosis"));
        selectedFeatures.add(RegionFeatures.Feature.fromLabel("StdDev"));
        selectedFeatures.add(RegionFeatures.Feature.fromLabel("Max"));
        selectedFeatures.add(RegionFeatures.Feature.fromLabel("Min"));

        // Define 3 classes ("No-Stained-No-Tumor", "Stained-Tumor" and "Background")
        final ArrayList<String> classes = new ArrayList<String>();
        classes.add("Background");
        classes.add( "No-Stained-No-Tumor" );
        classes.add( "Stained-Tumor" );

        // Define classifier
        RandomForest exampleClassifier = new RandomForest();
        TrainableSuperpixelSegmentation test =
                new TrainableSuperpixelSegmentation( inputImage, labelImage,
                        selectedFeatures, exampleClassifier, classes );
        //System.out.println(test.getFeaturesByRegion());

        // Define training regions
        int[] noStained = new int[]{ 176, 2111, 1322, 2298 };
        int[] stainedTum = new int[]{ 416, 591, 2013, 2024 };
        int[] background = new int[]{ 1, 360, 2742, 2795 };

        ArrayList<int[]> tags = new ArrayList<>();
        tags.add(background);
        tags.add(noStained);
        tags.add(stainedTum);

        // Train classifier using those labels
        if(test.trainClassifier(tags)){
            ImagePlus result = test.applyClassifier();
            result.show();
        }
        ImagePlus probs = test.getProbabilityMap();
        probs.show();
    }
}