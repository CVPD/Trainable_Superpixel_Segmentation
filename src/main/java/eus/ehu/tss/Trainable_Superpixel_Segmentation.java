/*
 * To the extent possible under law, the ImageJ developers have waived
 * all copyright and related or neighboring rights to this tutorial code.
 *
 * See the CC0 1.0 Universal license for details:
 *     http://creativecommons.org/publicdomain/zero/1.0/
 */

//Have just changed the name of the file

package eus.ehu.tss;

import ij.IJ;
import ij.ImagePlus;
import ij.WindowManager;
import ij.gui.ImageCanvas;
import ij.gui.StackWindow;
import ij.plugin.PlugIn;

import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;


public class Trainable_Superpixel_Segmentation implements PlugIn {

    private CustomWindow win;
    private ImagePlus inputImage;
    private ImagePlus supImage;
    private final ExecutorService exec = Executors.newFixedThreadPool(1);

    private class CustomWindow extends StackWindow
    {
        private Panel all = new Panel();
        private JPanel controlsPanel = new JPanel();
        private JPanel classifierPanel = new JPanel();
        private JPanel resultPanel = new JPanel();
        private JPanel classPanel = new JPanel();
        private JButton trainClassButton = null;
        private JButton loadClassButton = null;
        private JButton applyClassButton = null;
        private JButton settButton = null;
        private JButton plotButton = null;
        private JButton probButton = null;
        private JButton resButton = null;
        private JButton overlayButton = null;



        CustomWindow(ImagePlus imp)
        {
            super(imp, new ImageCanvas(imp));
            final ImageCanvas canvas = (ImageCanvas) getCanvas();
            GridBagLayout layout = new GridBagLayout();
            GridBagConstraints allConstraints = new GridBagConstraints();
            all.setLayout(layout);
            allConstraints.anchor = GridBagConstraints.CENTER;
            allConstraints.fill = GridBagConstraints.VERTICAL;
            allConstraints.gridwidth = 1;
            allConstraints.gridheight = 1;
            allConstraints.gridx = 0;
            allConstraints.gridy = 0;
            allConstraints.weightx = 0;
            allConstraints.weighty = 0;
            all.add(canvas,allConstraints);
            allConstraints.gridy++;

            //Control panel layout and constraints
            GridBagLayout controlLayout = new GridBagLayout();
            controlsPanel.setLayout(controlLayout);
            GridBagConstraints controlConstraints = new GridBagConstraints();
            controlConstraints.anchor = GridBagConstraints.CENTER;
            controlConstraints.fill = GridBagConstraints.HORIZONTAL;
            controlConstraints.gridwidth = 1;
            controlConstraints.gridheight = 1;
            controlConstraints.gridx = 0;
            controlConstraints.gridy = 0;
            controlConstraints.weightx = 0;
            controlConstraints.weighty = 0;

            //Classifier panel layout and constraints
            GridBagLayout classifierLayout = new GridBagLayout();
            classifierPanel.setLayout(classifierLayout);
            GridBagConstraints classifierConstraints = new GridBagConstraints();
            classifierConstraints.anchor = GridBagConstraints.WEST;
            classifierConstraints.fill = GridBagConstraints.VERTICAL;
            classifierConstraints.gridwidth = 1;
            classifierConstraints.gridheight = 1;
            classifierConstraints.gridx = 0;
            classifierConstraints.gridy = 0;
            classifierConstraints.weightx = 0;
            classifierConstraints.weighty = 0;

            //Result panel layout and constraints
            GridBagLayout resultLayout = new GridBagLayout();
            resultPanel.setLayout(resultLayout);
            GridBagConstraints resultConstraints = new GridBagConstraints();
            resultConstraints.anchor = GridBagConstraints.WEST;
            resultConstraints.fill = GridBagConstraints.VERTICAL;
            resultConstraints.gridwidth = 1;
            resultConstraints.gridheight = 1;
            resultConstraints.gridx = 0;
            resultConstraints.gridy = 0;
            resultConstraints.weightx = 0;
            resultConstraints.weighty = 0;

            //Classifier panel buttons
            trainClassButton = new JButton("Train classifier");
            classifierPanel.add(trainClassButton,classifierConstraints);
            classifierConstraints.gridy++;
            loadClassButton = new JButton("Load classifier");
            classifierPanel.add(loadClassButton,classifierConstraints);
            classifierConstraints.gridy++;
            applyClassButton = new JButton("Apply classifier");
            classifierPanel.add(applyClassButton,classifierConstraints);
            classifierConstraints.gridy++;
            settButton = new JButton("Settings");
            classifierPanel.add(settButton,classifierConstraints);
            classifierConstraints.gridy++;
            controlsPanel.add(classifierPanel,controlConstraints);
            controlConstraints.gridx++;

            //Result panel buttons
            plotButton = new JButton("Plot data");
            resultPanel.add(plotButton,resultConstraints);
            resultConstraints.gridy++;
            probButton = new JButton("Get probability");
            resultPanel.add(probButton,resultConstraints);
            resultConstraints.gridy++;
            resButton = new JButton("Create result");
            resultPanel.add(resButton,resultConstraints);
            resultConstraints.gridy++;
            overlayButton = new JButton("Toggle overlay");
            resultPanel.add(overlayButton,resultConstraints);
            resultConstraints.gridy++;
            controlsPanel.add(resultPanel,controlConstraints);
            controlConstraints.gridx++;

            all.add(controlsPanel,allConstraints);

            GridBagLayout wingb = new GridBagLayout();
            GridBagConstraints winc = new GridBagConstraints();
            winc.anchor = GridBagConstraints.CENTER;
            winc.fill = GridBagConstraints.BOTH;
            winc.weightx = 0;
            winc.weighty = 0;
            setLayout( wingb );
            add( all, winc );
            pack();
        }
    }

    @Override
    public void run(String s) {

        inputImage =IJ.openImage();
        supImage = IJ.openImage();
        if(inputImage == null ||supImage == null){
            IJ.error("Error when opening image");
        }else {
            win = new CustomWindow(inputImage);
        }

    }
    public static void main(String[] args){
        Class<?> clazz = Trainable_Superpixel_Segmentation.class;
        String url = clazz.getResource("/" + clazz.getName().replace('.', '/') + ".class").toString();
        String pluginsDir = url.substring("file:".length(), url.length() - clazz.getName().length() - ".class".length());
        System.setProperty("plugins.dir", pluginsDir);
        IJ.runPlugIn(clazz.getName(),"");

    }


}