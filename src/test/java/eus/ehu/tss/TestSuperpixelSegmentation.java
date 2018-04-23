package eus.ehu.tss;

import ij.IJ;
import ij.ImagePlus;
import ij.process.ImageProcessor;
import weka.classifiers.lazy.IBk;

import java.util.ArrayList;

public class TestSuperpixelSegmentation{
    public static void main(final String[] args){
        ImagePlus inputImage = IJ.openImage( TestSuperpixelSegmentation.class.getResource( "/grains3d.tif" ).getFile() );
        inputImage.show();
        ImagePlus labelImage = IJ.openImage( TestSuperpixelSegmentation.class.getResource( "/grains-catchment-basins3d.tif").getFile() );
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

        // Define 2 classes ("background" and "rice")
        final ArrayList<String> classes = new ArrayList<String>();
        classes.add( "background" );
        classes.add( "rice" );

        // Define classifier
        IBk exampleClassifier = new IBk();
        TrainableSuperpixelSegmentation test =
        		new TrainableSuperpixelSegmentation( inputImage, labelImage,
        				selectedFeatures, exampleClassifier, classes );
        System.out.println(test.getFeaturesByRegion());

        // Define training regions (one for background and 4 for rice grains)
        int[] rice = new int[4];
        rice[0] = 30; rice[1]=43; rice[2]=96;rice[3]=99;
        int[] background = new int[1];
        background[0] = 1;
        ArrayList<int[]> tags = new ArrayList<>();
        tags.add(background);
        tags.add(rice);

        // Train classifier using those labels
        if(test.trainClassifier(tags)){
            ImagePlus result = test.applyClassifier();
            result.show();
        }
        ImagePlus probs = test.getProbabilityMap();
        probs.show();
    }
}