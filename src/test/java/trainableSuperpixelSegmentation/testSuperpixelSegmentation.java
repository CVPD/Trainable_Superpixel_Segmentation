package trainableSuperpixelSegmentation;

import com.EHU.imagej.RegionFeatures;
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
        RegionFeatures regionFeatures = new RegionFeatures(inputImage,labelImage,selectedFeatures);
        System.out.println(regionFeatures.getMeanTable().toString());
        System.out.println(regionFeatures.getMedianTable().toString());
        System.out.println(regionFeatures.getModeTable().toString());
        System.out.println(regionFeatures.getSkewnessTable().toString());
        System.out.println(regionFeatures.getKurtosisTable().toString());
        System.out.println(regionFeatures.getStdDevTable().toString());
        System.out.println(regionFeatures.getMaxTable().toString());
        System.out.println(regionFeatures.getMinTable().toString());
    }
}