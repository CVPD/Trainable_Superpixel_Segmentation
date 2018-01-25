package trainableSuperpixelSegmentation;

import com.EHU.imagej.RegionFeatures;
import com.EHU.imagej.TrainableSuperpixelSegmentation;
import ij.IJ;
import ij.ImagePlus;

import java.util.ArrayList;

public class testSuperpixelSegmentation{
    public static void main(final String[] args){
        ImagePlus inputImage = IJ.openImage();
        inputImage.show();
        ImagePlus labelImage = IJ.openImage();
        labelImage.show();
        ArrayList<RegionFeatures.Feature> selectedFeatures = new ArrayList<RegionFeatures.Feature>();
        selectedFeatures.add(RegionFeatures.Feature.fromLabel("Mean"));
        selectedFeatures.add(RegionFeatures.Feature.fromLabel("Median"));
        selectedFeatures.add(RegionFeatures.Feature.fromLabel("Mode"));
        selectedFeatures.add(RegionFeatures.Feature.fromLabel("Skewness"));
        selectedFeatures.add(RegionFeatures.Feature.fromLabel("Kurtosis"));
        selectedFeatures.add(RegionFeatures.Feature.fromLabel("StdDev"));
        selectedFeatures.add(RegionFeatures.Feature.fromLabel("Max"));
        selectedFeatures.add(RegionFeatures.Feature.fromLabel("Min"));
        ArrayList<Integer> temp = new ArrayList<Integer>();
        TrainableSuperpixelSegmentation test = new TrainableSuperpixelSegmentation(inputImage,labelImage,temp, selectedFeatures);
        test.showFeaturesByRegion();
    }
}