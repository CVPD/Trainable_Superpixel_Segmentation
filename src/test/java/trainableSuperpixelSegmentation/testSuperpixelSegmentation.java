package trainableSuperpixelSegmentation;

import com.EHU.imagej.RegionFeatures;
import ij.IJ;
import ij.ImagePlus;
import ij.measure.ResultsTable;
import inra.ijpb.measure.IntensityMeasures;

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
        ArrayList<ResultsTable> results = new ArrayList<ResultsTable>();
        final IntensityMeasures im = new IntensityMeasures( inputImage, labelImage );
        results.add( im.getMean() );
        results.add( im.getStdDev() );
        results.add( im.getMax() );
        results.add( im.getMin() );
        results.add( im.getMedian() );
        results.add( im.getMode() );
        results.add( im.getSkewness() );
        results.add( im.getKurtosis() );
        results.add( im.getNumberOfVoxels() );
        results.add( im.getVolume() );

        ResultsTable mergedTable = new ResultsTable();
        final int numLabels = results.get( 0 ).getCounter();

        for(int i=0; i < numLabels; i++ )
        {
            mergedTable.incrementCounter();
            String label = results.get( 0 ).getLabel( i );
            mergedTable.addLabel(label);

            for( int j=0; j<results.size(); j++ )
            {
                String measure = results.get( j ).getColumnHeading( 0 );
                double value = results.get( j ).getValue( measure, i );
                mergedTable.addValue( measure, value );
            }
        }

        mergedTable.show( inputImage.getShortTitle() +
                "-intensity-measurements" );
    }
}