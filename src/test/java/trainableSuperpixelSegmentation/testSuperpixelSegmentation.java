package trainableSuperpixelSegmentation;

import eus.ehu.superpixel.learning.supervised.RegionFeatures;
import eus.ehu.superpixel.learning.supervised.TrainableSuperpixelSegmentation;
import ij.IJ;
import ij.ImagePlus;
import weka.classifiers.lazy.IBk;

import java.util.ArrayList;

public class testSuperpixelSegmentation{
    public static void main(final String[] args){
        ImagePlus inputImage = IJ.openImage("D:\\Proiektuak\\TFG\\Trainable_Superpixel_Segmentation\\src\\test\\resources\\grains.png");
        inputImage.show();
        ImagePlus labelImage = IJ.openImage("D:\\Proiektuak\\TFG\\Trainable_Superpixel_Segmentation\\src\\test\\resources\\grains-catchment-basins.png");
        labelImage.show();
        ArrayList<RegionFeatures.Feature> selectedFeatures = new ArrayList<>();
        selectedFeatures.add(RegionFeatures.Feature.fromLabel("Mean"));
        selectedFeatures.add(RegionFeatures.Feature.fromLabel("Median"));
        selectedFeatures.add(RegionFeatures.Feature.fromLabel("Mode"));
        selectedFeatures.add(RegionFeatures.Feature.fromLabel("Skewness"));
        selectedFeatures.add(RegionFeatures.Feature.fromLabel("Kurtosis"));
        selectedFeatures.add(RegionFeatures.Feature.fromLabel("StdDev"));
        selectedFeatures.add(RegionFeatures.Feature.fromLabel("Max"));
        selectedFeatures.add(RegionFeatures.Feature.fromLabel("Min"));

        TrainableSuperpixelSegmentation test = new TrainableSuperpixelSegmentation(inputImage,labelImage, selectedFeatures);
        test.showFeaturesByRegion();
        IBk exampleClassifier = new IBk();
        int[] rice = new int[4];
        rice[0] = 30; rice[1]=43; rice[2]=96;rice[3]=99;
        int[] background = new int[1];
        background[0] = 1;
        ArrayList<int[]> tags = new ArrayList<>();
        tags.add(background); tags.add(rice);
        test.trainClassifier(exampleClassifier,tags);
        test.applyClassifier();
    }
}