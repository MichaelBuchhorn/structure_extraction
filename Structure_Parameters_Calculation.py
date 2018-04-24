# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import math


def calculateAggregatedSpectrum(spectrumToAnalyse, nonAggregatedSpectrum, index1, index2):
	if(len(spectrumToAnalyse) != len(nonAggregatedSpectrum)):
		raise AssertionError("Size of array that contain the spectra not identical.")
		
	scalingFactor = np.sum(spectrumToAnalyse[index1:index2]) / np.sum(nonAggregatedSpectrum[index1:index2])
	aggregatedSpectrum = spectrumToAnalyse - nonAggregatedSpectrum * scalingFactor
	
	return aggregatedSpectrum, scalingFactor


def calculateAggregatedSpectra_bulk(spectrumToAnalyse, index1, index2, nonAggregatedSpectrum = None):
	aggregatedSpectra = np.zeros((0, len(spectrumToAnalyse[0])))
	scalingFactors = np.zeros((0))
	
	if (nonAggregatedSpectrum is None):
		nonAggregatedSpectrum = spectrumToAnalyse[0] 
	
	for i in range(0, len(spectrumToAnalyse)):
		aggregatedSpectrum_, scalingFactor_  = calculateAggregatedSpectrum(spectrumToAnalyse[i], nonAggregatedSpectrum, index1, index2)
		aggregatedSpectra = np.vstack((aggregatedSpectra, aggregatedSpectrum_))
		scalingFactors = np.append(scalingFactors, scalingFactor_)

	return aggregatedSpectra, scalingFactors


def calculateFractionOfAggregation(spectrumToAnalyse, nonAggregatedSpectrum, index1, index2, f_factor = -1.0):
	
	aggregatedSpectrum, scalingFactor = calculateAggregatedSpectrum(spectrumToAnalyse, nonAggregatedSpectrum, index1, index2)
	
	if(f_factor < 0.0):
		#Sum has to be potentially change into a real integral along the energy axis
		f_factor = np.sum(aggregatedSpectrum) / np.sum(nonAggregatedSpectrum * (1.0 - scalingFactor))
	fractionOfAggregation = np.sum(aggregatedSpectrum) / np.sum(spectrumToAnalyse) / f_factor
	
	return fractionOfAggregation, aggregatedSpectrum, scalingFactor


def calculateFractionOfAggregation_bulk(absorptionSpectra, index1, index2, nonAggregatedSpectrum = None, f_factor = -1.0):
	aggregatedSpectra = np.zeros((0, len(absorptionSpectra[0])))
	fractionOfAggregation = np.zeros((0))
	scalingFactors = np.zeros((0))
	
	if (nonAggregatedSpectrum is None):
		nonAggregatedSpectrum = absorptionSpectra[0] 
	
	for i in range(0, len(absorptionSpectra)):
		fractionOfAggregation_, aggregatedSpectrum_, scalingFactor_  = calculateFractionOfAggregation(absorptionSpectra[i], nonAggregatedSpectrum, index1, index2, f_factor)
		fractionOfAggregation = np.append(fractionOfAggregation, fractionOfAggregation_)
		aggregatedSpectra = np.vstack((aggregatedSpectra, aggregatedSpectrum_))
		scalingFactors = np.append(scalingFactors, scalingFactor_)

	return fractionOfAggregation, aggregatedSpectra, scalingFactors


def calculateFreeExcitionBandwidth_eV(peakRatio_A00_A01, vibrationalEnergy_eV, refractionIndexRatio_n00_n01 = 1.0):
	a = math.sqrt(peakRatio_A00_A01 / refractionIndexRatio_n00_n01)
	return (vibrationalEnergy_eV - a * vibrationalEnergy_eV) / (0.073 * a + 0.24)

def spectrumSum_bulk(spectrum):
	summedUpSpectrum = np.zeros((0))
	for i in range(0, len(spectrum)):
		summedUpSpectrum = np.append(summedUpSpectrum, np.sum(spectrum[i]))
		
	return summedUpSpectrum

def interpolateNaNValues(array):
	indices = np.arange(array.shape[0])
	notNaNValues = np.where(np.isfinite(array))
	interpolatedArray = interpolate.interp1d(indices[notNaNValues], array[notNaNValues], bounds_error = False, fill_value = "extrapolate")
	return np.where(np.isfinite(array), array, interpolatedArray(indices))

def addAdjacentSpectra(spectra):
	for i in range(0, len(spectra)-1):
		spectra[i] += spectra[i+1]
	
	return np.delete(spectra, len(spectra)-1, 0)


def loadAbsorptionSpectra(path, referenceScalingFactor = 1.0):
	absorption = np.loadtxt(path + '/Absorption.dat')
	background = np.loadtxt(path + '/Background.dat')
	reference = np.loadtxt(path + '/Reference.dat')
	
	absorption -= background
	absorption = addAdjacentSpectra(absorption)
	absorption /= reference * referenceScalingFactor
	
	# Ignore warning that there are invalid values when caluclation the OD
	# The invalid values will be interpolated later using interpolateNaNValues()
	np.seterr(all='ignore')
	calculateOD = lambda t: -np.log10(t)
	calculateOD_vectorized = np.vectorize(calculateOD)
	absorption = calculateOD_vectorized(absorption)
	
	for i in range(0, len(absorption)):
		absorption[i] = interpolateNaNValues(absorption[i])
	
	return absorption


def loadPLSpectra(path):
	pl = np.loadtxt(path + '/PL.dat')
	background = np.loadtxt(path + '/Background.dat')
	
	pl -= background
	pl = addAdjacentSpectra(pl)

	return pl



absorptionSpectra = loadAbsorptionSpectra('C:/Users/MAB/Desktop', 1.88)
for i in range(0, len(absorptionSpectra)):
	absorptionSpectra[i] -= np.average(absorptionSpectra[i][800:920])
absorptionSpectra = absorptionSpectra[11:80,230:820] #get ROI

plt.title('Absorption Spectrum')
plt.imshow(absorptionSpectra, interpolation='bilinear', origin='lower', aspect=len(absorptionSpectra[0])/len(absorptionSpectra), vmax=absorptionSpectra.max(), vmin=0.0)
plt.show()

fractionOfAggregation, aggregatedSpectra, scalingFactors = calculateFractionOfAggregation_bulk(absorptionSpectra, 0, 50, f_factor = 1.4)

plt.title('Aggregated Spectrum')
plt.imshow(aggregatedSpectra, interpolation='bilinear', origin='lower', aspect=len(aggregatedSpectra[0])/len(aggregatedSpectra), vmax=aggregatedSpectra.max(), vmin=0.0)
plt.show()

plt.title('Fraction of Aggregation')
plt.plot(fractionOfAggregation)
plt.show()


plSpectra = loadPLSpectra('C:/Users/MAB/Desktop')
for i in range(0, len(plSpectra)):
	plSpectra[i] -= np.average(plSpectra[i][0:250])
plSpectra = plSpectra[11:80,450:950] #get ROI

plt.title('PL Spectrum')
plt.imshow(plSpectra, interpolation='bilinear', origin='lower', aspect=len(plSpectra[0])/len(plSpectra), vmax=plSpectra.max(), vmin=0.0)
plt.show()

aggregatedSpectra, scalingFactors = calculateAggregatedSpectra_bulk(plSpectra, 30, 125)
plt.imshow(aggregatedSpectra, interpolation='bilinear', origin='lower', aspect=len(aggregatedSpectra[0])/len(aggregatedSpectra), vmax=aggregatedSpectra.max(), vmin=0.0)
plt.show()

spectrumSum = spectrumSum_bulk(plSpectra)
plt.plot(spectrumSum/spectrumSum.max())
plt.show()

spectrumIndex = 35
wavelength = np.loadtxt('C:/Users/MAB/Desktop/Wavelengths.dat')[450:950]
plt.plot(wavelength, plSpectra[spectrumIndex], label="Test Spectrum")
plt.plot(wavelength, aggregatedSpectra[spectrumIndex], label="Aggregated")
plt.plot(wavelength, plSpectra[0] * scalingFactors[spectrumIndex], label="Fitted Non-Aggregated")
plt.xlabel("Wellenlänge [nm]")
plt.ylabel("PL intensity [a.u.]")
plt.legend()
plt.show()