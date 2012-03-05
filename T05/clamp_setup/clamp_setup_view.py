#-------------------------------------------------------------------------------
#
# Copyright (c) 2009, IMB, RWTH Aachen.
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in simvisage/LICENSE.txt and may be redistributed only
# under the conditions described in the aforementioned license.  The license
# is also available online at http://www.simvisage.com/licenses/BSD.txt
#
# Thanks for using Simvisage open source!
#
# Created on Sep 21, 2009 by: rrypl

from enthought.traits.api import Instance, on_trait_change, \
                                 Event
from enthought.traits.ui.api import \
    View, Item, VGroup, ModelView, HSplit, VSplit
from enthought.traits.ui.menu import OKButton

from numpy import linspace, frompyfunc, zeros, column_stack, argwhere, \
                 abs, array, hstack, cos

from clamp_setup_model import ClampSetup
from clamp_const_model import ClampConst
from util.traits.editors.mpl_figure_editor import MPLFigureEditor
from matplotlib.figure import Figure

#--------------------------------------------------------------------------
# MODEL_VIEW
#--------------------------------------------------------------------------
class CSModelView ( ModelView ):

    model = Instance( ClampSetup )
    def _model_default( self ):
        return ClampSetup()

    figure = Instance( Figure )
    def _figure_default( self ):
        figure = Figure( facecolor = 'white' )
        figure.add_axes( [0.08, 0.13, 0.85, 0.74] )
        return figure

    data_changed = Event

    @on_trait_change( 'model.+modified' )
    def refresh( self ):

        figure = self.figure
        axes = figure.gca()
        axes.clear()

        cs = self.model
        xu, u, xe, e = cs.get_values()

        axes.set_xlabel( 'position on yarn', weight = 'semibold' )
        axes.set_axis_bgcolor( color = 'white' )
        axes.ticklabel_format( scilimits = ( -3., 4. ) )
        axes.grid( color = 'gray', linestyle = '--', linewidth = 0.5, alpha = 0.7 )
        if cs.switch == 'strains':
            axes.plot( xe, e, lw = 2, color = 'blue' )
            axes.set_ylabel( 'strains [-]', weight = 'semibold' )
            axes.plot( array( [cs.l1, cs.l1] ), array( [0, max( e )] ), lw = 2, color = 'red', linestyle = 'dashed' )
            axes.plot( array( [cs.l1 + cs.L, cs.l1 + cs.L] ), array( [0, max( e )] ), lw = 2, color = 'red', linestyle = 'dashed' )
            axes.text( cs.l1 + cs.L / 2, max( e ) * 0.95, 'clamp', size = 'large', ha = 'center' )
            axes.text( cs.l1 + cs.L + cs.lt / 2, max( e ) * 0.95, 'tested length', size = 'large', ha = 'center' )
        elif cs.switch == 'displacement':
            axes.plot( xu, u, lw = 2, color = 'blue' )
            axes.set_ylabel( 'displacement [m]', weight = 'semibold' )
            axes.plot( array( [cs.l1, cs.l1] ), array( [0, max( u )] ), lw = 2, color = 'red', linestyle = 'dashed' )
            axes.plot( array( [cs.l1 + cs.L, cs.l1 + cs.L] ), array( [0, max( u )] ), lw = 2, color = 'red', linestyle = 'dashed' )
            axes.text( cs.l1 + cs.L / 2, max( u ) * 0.95, 'clamp', size = 'large', ha = 'center' )
            axes.text( cs.l1 + cs.L + cs.lt / 2, max( u ) * 0.95, 'tested length', size = 'large', ha = 'center' )
        else:
            print 'NOT YET IMPLEMENTED'
#            axes.plot(xslip, yslip, lw = 2, color = 'blue')
#            axes.set_ylabel('slip in clamp [m]', weight = 'semibold')
#            axes.set_xlabel('position in clamp [mm]', weight = 'semibold')

        self.data_changed = True

    traits_view = View( 
                   HSplit( 
                       VGroup( 
                             Item( 'model@', show_label = False, resizable = True ),
                             label = 'Material parameters',
                             id = 'cs.viewmodel.model',
                             dock = 'tab',
                             ),
                    VSplit( 
                       VGroup( 
                               Item( 'figure',
                                     editor = MPLFigureEditor(),
                                     resizable = True, show_label = False ),
                                label = 'yarn displacement',
                               id = 'cs.viewmodel.plot_window',
                               dock = 'tab',
                             ),
                             id = 'cs.viewmodel.right',
                            ),
                        id = 'cs.viewmodel.splitter',
                    ),
                    title = 'tested yarn',
                    id = 'cs.viewmodel',
                    dock = 'tab',
                    resizable = True,
                    buttons = [OKButton],
                    height = 0.8, width = 0.8
                    )


if __name__ == '__main__':
    c = CSModelView( model = ClampSetup() )
    c.refresh()
    c.configure_traits()
