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


public class Trainable_Superpixel_Segmentation implements PlugIn {

    private CustomWindow win;
    private ImagePlus inputImage;
    private ImagePlus supImage;


    private class CustomWindow extends StackWindow
    {
        private Panel all = new Panel();
        private JPanel buttonsPanel = new JPanel();
        private JButton clusterizeButton = null;
        CustomWindow(ImagePlus imp)
        {
            super(imp, new ImageCanvas(imp));
            final ImageCanvas canvas = (ImageCanvas) getCanvas();
            GridBagLayout layout = new GridBagLayout();
            GridBagConstraints allConstraints = new GridBagConstraints();
            all.setLayout(layout);
            allConstraints.anchor = GridBagConstraints.CENTER;
            allConstraints.fill = GridBagConstraints.BOTH;
            allConstraints.gridwidth = 2;
            allConstraints.gridheight = 1;
            allConstraints.gridx = 0;
            allConstraints.gridy = 0;
            allConstraints.weightx = 0;
            allConstraints.weighty = 0;
            all.add(canvas,allConstraints);
            allConstraints.gridy++;

            clusterizeButton = new JButton("Clusterize");
            buttonsPanel.add(clusterizeButton);
            all.add(buttonsPanel,allConstraints);


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