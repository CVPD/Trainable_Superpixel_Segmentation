package eus.ehu.tss;

import ij.IJ;
import ij.ImagePlus;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;

import java.util.ArrayList;


public class TestColorSuperpixelSegmentation {
    public static void main(final String[] args){
        ImagePlus inputImage = IJ.openImage( TestSuperpixelSegmentation.class.getResource( "/eval/histogram-matched-TMA/TMA-01.png" ).getFile() );
        inputImage.show();
        ImagePlus labelImage = IJ.openImage( TestSuperpixelSegmentation.class.getResource( "/eval/superpixels/SLIC-01.zip").getFile() );
        labelImage.show();
        ImagePlus gt = IJ.openImage( TestSuperpixelSegmentation.class.getResource( "/eval/groundtruth/groundtruth-01.png").getFile() );
        gt.show();
        ImagePlus testImage = IJ.openImage( TestSuperpixelSegmentation.class.getResource("/eval/histogram-matched-TMA/TMA-09.png").getFile());
        ImagePlus labelTest = IJ.openImage( TestSuperpixelSegmentation.class.getResource( "/eval/superpixels/SLIC-09.zip").getFile() );


        // Use all features
        ArrayList<RegionFeatures.Feature> selectedFeatures = new ArrayList<>();
        String[] selectedFs = RegionFeatures.Feature.getAllLabels();
        for(int i=0;i<selectedFs.length;++i){
            selectedFeatures.add(RegionFeatures.Feature.fromLabel(selectedFs[i]));
        }

        // Define 3 classes ("No-Stained-No-Tumor", "Stained-Tumor" and "Background")
        final ArrayList<String> classes = new ArrayList<String>();
        classes.add("Background");
        classes.add( "No-Stained-No-Tumor" );
        classes.add( "Stained-Tumor" );

        // Define classifier
        RandomForest exampleClassifier = new RandomForest();
        TrainableSuperpixelSegmentation tss  = new TrainableSuperpixelSegmentation();
        //System.out.println(test.getFeaturesByRegion());
        Instances training = RegionColorFeatures.calculateLabeledColorFeatures(inputImage,labelImage,gt,selectedFeatures,classes);
        // Train classifier using those labels
        J48 classifier = new J48();
        try {
            classifier.buildClassifier(training);
            tss.setClassifier(classifier);
            tss.setClassifierTrained(true);
            tss.setSelectedFeatures(selectedFeatures);
            tss.setClasses(classes);
            tss.setInputImage(testImage);
            tss.setLabelImage(labelTest);
            tss.calculateRegionFeatures();
            ImagePlus result = tss.applyClassifier();
            result.show();
            IJ.save(result,"D:/Documents");
        } catch (Exception e) {
            e.printStackTrace();
        }

    }
}