package eus.ehu.tss;

import ij.IJ;
import ij.ImagePlus;
import weka.classifiers.lazy.IBk;

import java.util.ArrayList;


public class TestColorSuperpixelSegmentation {
    public static void main(final String[] args){
        ImagePlus inputImage = IJ.openImage( TestSuperpixelSegmentation.class.getResource( "/TMA.png" ).getFile() );
        inputImage.show();
        ImagePlus labelImage = IJ.openImage( TestSuperpixelSegmentation.class.getResource( "/TMA-Segmentation.png").getFile() );
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
        IBk exampleClassifier = new IBk();
        TrainableSuperpixelSegmentation test =
                new TrainableSuperpixelSegmentation( inputImage, labelImage,
                        selectedFeatures, exampleClassifier, classes );
        System.out.println(test.getFeaturesByRegion());

        // Define training regions
        int[] noStained = new int[4];
        noStained[0] = 38; noStained[1]=108; noStained[2]=163;noStained[3]=223;
        int[] stainedTum = new int[4];
        stainedTum[0] = 34; stainedTum[1]=60;stainedTum[2]=181;stainedTum[3]=173;
        int[] background = new int[4];
        background[0]=1;background[1]=211;background[2]=35;background[3]=226;
        ArrayList<int[]> tags = new ArrayList<>();
        tags.add(background);
        tags.add(noStained);
        tags.add(stainedTum);

        // Train classifier using those labels
        if(test.trainClassifier(tags)){
            ImagePlus result = test.applyClassifier();
            result.show();
        }
    }
}