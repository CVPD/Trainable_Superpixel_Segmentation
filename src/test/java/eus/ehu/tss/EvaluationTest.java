package eus.ehu.tss;


import ij.IJ;
import ij.ImagePlus;
import weka.classifiers.AggregateableEvaluation;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;

import java.io.*;
import java.util.ArrayList;
import java.util.zip.GZIPOutputStream;

public class EvaluationTest {
    public static void main(final String[] args) {
        System.out.println("Open input image");
        ImagePlus inImage = IJ.openImage();
        System.out.println("Open superpixel image");
        ImagePlus spImage = IJ.openImage();
        System.out.println("Open groundtruth image");
        ImagePlus gtImage = IJ.openImage();
        final ArrayList<String> classes = new ArrayList<String>();
        classes.add("Background");
        classes.add( "No-Stained-No-Tumor" );
        classes.add( "Stained-Tumor" );
        ArrayList<RegionFeatures.Feature> selectedFeatures = new ArrayList<>();
        String[] selectedFs = RegionFeatures.Feature.getAllLabels();
        for(int i=0;i<selectedFs.length;++i){
            selectedFeatures.add(RegionFeatures.Feature.fromLabel(selectedFs[i]));
        }
        Instances trainingData = RegionColorFeatures.calculateLabeledColorFeatures(inImage,spImage,gtImage,selectedFeatures,classes);
        System.out.println("Done, number of calculated instances: "+trainingData.numInstances());
    }
}
